"""
Yolo V5 工地安全检测：
检测工人， 安全帽， 安全衣， 并且基于 IOU, 检测工人是否佩戴安全帽和安全衣

FPS ~~ 55
yolov5_s:  map(iou=0.5) 0.88
yolov5_m: map (iou=0.5) 0.86
yolov5_n: map (iou=0.5) 0.67

"""

import torch
import cv2
import time

class detection:
    def __init__(self):
        # 加载模型
        self.model = torch.hub.load('./yolov5', 'custom', path='./yolov5/yolo_test/base_m2/weights/best.pt', source='local')
        self.model.conf = 0.4  # NMS IOU threshold

        # 摄像头
        self.cap = cv2.VideoCapture(0)


    def get_iou(self,boxA, boxB):
        """
        计算两个框的IOU

        @param: boxA,boxB list形式的框坐标
        @return: iou float
        """
        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def get_person_info_list(self, person_list, hat_list, vest_list):
        """
        获取每个人的完整信息

        @param: person_list,hat_list,vest_list numpy array
        @return  person_info_list list
        """
        hat_iou_thresh = 0
        vest_iou_thresh = 0

        person_info_list = []

        for person in person_list:
            person_info_item = [[], [], []]
            # 人体框
            person_box = person[:5]  # (xmin,ymin,xmax,ymax, confidence)

            person_info_item[0] = person_box
            # 依次与帽子计算IOU
            for hat in hat_list:
                hat_box = hat[:6]  # (xmin,ymin,xmax,ymax, confidence)
                hat_iou = self.get_iou(person_box, hat_box)

                if hat_iou > hat_iou_thresh:
                    person_info_item[1] = hat_box
                    break

            # 依次与防护服计算IOU
            for vest in vest_list:
                vest_box = vest[:5]  # (xmin,ymin,xmax,ymax, confidence)
                vest_iou = self.get_iou(person_box, vest_box)

                if vest_iou > vest_iou_thresh:
                    person_info_item[2] = vest_box
                    break

            person_info_list.append(person_info_item)

        return person_info_list  # numpy, List(list())

    def draw(self, frame, person_info_list):
        '''
        图片上绘制结果
        '''
        for person in person_info_list:
            person_box = person[0]
            hat_box = person[1]
            vest_box = person[2]

            if len(person_box) > 0:
                p_l, p_t, p_r, p_b = person_box[:4].astype('int')
                conf = person_box[4]

                conf_txt = str(round(conf * 100, 1)) + '%'

                cv2.rectangle(frame, (p_l, p_t), (p_r, p_b), (0, 255, 0), 5)
                cv2.putText(frame, conf_txt, (p_l, p_t - 35), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

            if len(hat_box) > 0:
                l, t, r, b = hat_box[:4].astype('int')

                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 5)

            if len(vest_box) > 0:
                l, t, r, b = vest_box[:4].astype('int')
                conf = vest_box[4]

                cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 5)
                conf_txt = str(round(conf * 100, 1)) + '%'
                cv2.putText(frame, conf_txt, (l, t - 35), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)


    def detect(self):
        # 获取摄像头每一帧
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            # 镜像翻转
            frame = cv2.flip(frame, 1)
            # BGR--> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            # 推理
            results = self.model(frame_rgb)
            pd = results.pandas().xyxy[0]

            person_list = pd[pd['name'] == 'person'].to_numpy()
            vest_list = pd[pd['name'] == 'vest'].to_numpy()
            hat_list = pd[pd['name'].str.contains('helmet')].to_numpy()

            # 获取人员信息
            person_info_list = self.get_person_info_list(person_list, hat_list, vest_list)

            # 绘制结果
            self.draw(frame, person_info_list)

            end_time = time.time()
            fps = 1.0 / (end_time-start_time)

            cv2.putText(frame, 'FPS: ' + str(round(fps, 2)), (30, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Person: ' + str(len(person_info_list)), (30, 100), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

            # 显示画面
            cv2.imshow("Detection", frame)
            # 退出条件
            if cv2.waitKey(10) & 0xFF == 27:
                break

        # 释放
        self.cap.release()
        cv2.destroyAllWindows()


det = detection()
det.detect()