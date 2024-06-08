from torch import nn
import torch
import yaml
import numpy as np
from pathlib import Path
import cv2
import torchvision
import pandas as pd
import os
import xlsxwriter

class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

def yaml_load(file="data.yaml"):
    """Safely loads and returns the contents of a YAML file specified by `file` argument."""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    """Validates if a file or files have an acceptable suffix, raising an error if not."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initializes an ensemble of models to be used for aggregated predictions."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs forward pass aggregating outputs from an ensemble of models.."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self,
            weights="yolov5s.pt",
            device=torch.device("cuda:0"),
            dnn=False,
            data=None,
            fp16=False,
            imgsz=(480, 640, 3),
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=5,
            agnostic_nms=False,
            hide_conf=False):
        """Initializes DetectMultiBackend with support for various inference backends, including PyTorch and ONNX."""
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt = True
        fp16 &= pt  # FP16
        stride = 32  # default stride
        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=False)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        colors = Colors()
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        self.__dict__.update(locals())  # assign all variables to self
        # warm-up
        # warmup_types = onnx, pt
        raw_img = np.empty(imgsz, dtype=np.half if fp16 else np.float_)  # input
        for _ in range(1):  #
            self.forward(raw_img)  # warmup

    def forward(self, raw_img):
        # preprocess
        im = self.letterbox(raw_img, new_shape=(480, 640), stride=32, auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).unsqueeze(0).to(self.device)
        im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        """Performs YOLOv5 inference on input images with options for augmentation and visualization."""
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.pt:  # PyTorch
            y = self.model(im)

        if isinstance(y, (list, tuple)):
            pred = self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            pred = self.from_numpy(y)

        pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres, self.agnostic_nms, max_det=self.max_det)
        # Process predictions
        ret_det = {'label': list(), 'confidence': list(), 'xyxy': list()}
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = self.scale_boxes(im.shape[2:], det[:, :4], raw_img.shape).round()
                # conf_temp = 0
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    # label = (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    ret_det['label'].append(self.names[c])
                    ret_det['confidence'].append(f'{conf:.2f}')
                    ret_det['xyxy'].append(xyxy)
        return ret_det

    def from_numpy(self, x):
        """Converts a NumPy array to a torch tensor, maintaining device compatibility."""
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def box_iou(self, box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.

        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def xywh2xyxy(self, x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def non_max_suppression(self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nm=0,  # number of masks
    ):
        """
        Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = x[i]
        return output

    def clip_boxes(self, boxes, shape):
        """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes

    def box_label(self, im, box, label, color, txt_color=(255, 255, 255)):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(im, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
        if label:
            w, h = cv2.getTextSize(label, 0, fontScale=1.0, thickness=2)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                im,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                1.0,
                txt_color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    def letterbox(self, im, new_shape=(640, 480), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

def box_label(im, box, label, color, txt_color=(255, 255, 255)):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=1.0, thickness=2)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            im,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            1.0,
            txt_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return im

if __name__ == '__main__':
    color = Colors()
    model_barrier = DetectMultiBackend(
            weights="./weights/barrier.pt",
            device=torch.device("cuda:0"),
            dnn=False,
            data=None,
            fp16=False,
            imgsz=(640, 640, 3),
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=5,
            agnostic_nms=False,
            hide_conf=False)
    model_coast = DetectMultiBackend(
        weights="./weights/coast.pt",
        device=torch.device("cuda:0"),
        dnn=False,
        data=None,
        fp16=False,
        imgsz=(640, 640, 3),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=5,
        agnostic_nms=False,
        hide_conf=False)
    root_path = 'B:/C3/datasets/Barrier/images/val'
    res_save = {'label': list(), 'img_id': list(), 'confidence': list(), 'xmin': list(), 'ymin': list(), 'xmax': list(), 'ymax': list()}
    for img_name in os.listdir(root_path):
        img = cv2.imread(os.path.join(root_path, img_name))
        res_barrier = model_barrier(img)
        res_coast = model_coast(img)
        for label, confidence, xyxy in zip(res_barrier['label'], res_barrier['confidence'], res_barrier['xyxy']):
            res_save['label'].append(label)
            res_save['img_id'].append(img_name)
            res_save['confidence'].append(confidence)
            res_save['xmin'].append(int(xyxy[0]))
            res_save['ymin'].append(int(xyxy[1]))
            res_save['xmax'].append(int(xyxy[2]))
            res_save['ymax'].append(int(xyxy[3]))
            box_label(img, xyxy, label + confidence, color(0, True))
        for label, confidence, xyxy in zip(res_coast['label'], res_coast['confidence'], res_coast['xyxy']):
            res_save['label'].append(label)
            res_save['img_id'].append(img_name)
            res_save['confidence'].append(confidence)
            res_save['xmin'].append(int(xyxy[0]))
            res_save['ymin'].append(int(xyxy[1]))
            res_save['xmax'].append(int(xyxy[2]))
            res_save['ymax'].append(int(xyxy[3]))
            box_label(img, xyxy, label + confidence, color(0, True))
        cv2.imwrite(os.path.join('./ztest_img', img_name), img)
        # break

    # save_excel
    df = pd.DataFrame(res_save)
    df.to_csv(os.path.join('1.csv'), index=False)

    # with pd.ExcelWriter(os.path.join('1.csv'), engine='xlsxwriter') as writer:
    #     df.to_excel(writer, sheet_name='1', index=False)
    #     # 定义表头的格式：加粗和背景颜色
    #     workbook = writer.book
    #     worksheet = writer.sheets['1']
    #
    #     header_format = workbook.add_format({
    #         'bold': False,
    #         'border': 0
    #     })
    #     # 设置表头格式
    #     for col_num, value in enumerate(df.columns.values):
    #         worksheet.write(0, col_num, value, header_format)
    #     # 设置单元格宽度
    #     worksheet.set_column('A:G', 20)