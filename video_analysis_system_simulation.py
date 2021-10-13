import _init_paths
import argparse
import darknet
import logging
import time
import os
import sys
import threading
import numpy as np
import scheduler
from cv2 import cv2 
from queue import Queue
from typing import List

# w = 0
# max_segments = 0
control_queue = Queue(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Video Analysis Simulation System")
    parser.add_argument("--input", type=str, required=True,
                        help="Video source config file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file.")
    # parser.add_argument("--classname", type=str, required=True,
    #                     help="class.")
    parser.add_argument("--thresh", type=float, default=0.6,
                        help="Detection threshhold.")
    return parser.parse_args()

def initialize():
    w = 1
    t = 1
    max_segments = 600
    iou_threshold = 0.7
    capacity = 9000
    num_of_flows = len(video_processor_list)
    scheduler.MYSolution.init(t,len(video_processor_list),capacity,[0.7 for i in range(len(video_processor_list))])
    # scheduler.PeriodicMethod.init(t,len(video_processor_list),capacity,[0.7 for i in range(len(video_processor_list))])
    # scheduler.OneTimeMethod.init(t,len(video_processor_list),capacity,[0.7 for i in range(len(video_processor_list))])
    return t, w, max_segments, iou_threshold, num_of_flows

def resizebbox(dets, ow, oh, nw, nh):
    d = []
    for det in dets:
        bbox = list(det[2])
        bbox[0] = bbox[0] * nw / ow
        bbox[1] = bbox[1] * nh / oh
        bbox[2] = bbox[2] * nw / ow
        bbox[3] = bbox[3] * nh / oh
        d.append((det[0],det[1],tuple(bbox)))
    return tuple(d)

def video_processing(cap, msg_queue, det_queue, network, class_names):
    thread_name = threading.currentThread().getName()

    # 考虑可能同时会执行好几个配置
    configuration_list = [(10,608)]
    frame_lists = [frame_processing_list(i[0]) for i in configuration_list]
    darknet_images = [darknet.make_image(i[1], i[1], 3) for i in configuration_list]

    default_resolution = 608
    default_img = darknet.make_image(default_resolution, default_resolution, 3)

    detections = []
    cur_t = 0
    cur_frame = 0
    while cap.isOpened() or not msg_queue.empty():
        logger.debug('{} waiting for frame {} of second {}. Queue empty?{}'.\
            format(thread_name, cur_frame, cur_t, msg_queue.empty()))
        msg = msg_queue.get()
        # logger.warning('{}.'.format(msg[0]))
        logger.debug('{} got frame {} of second {}. Queue empty?{}'.\
            format(thread_name, cur_frame, cur_t, msg_queue.empty()))
        if msg[0] == 'configuration_list':
            configuration_list = msg[1]
            frame_lists = [frame_processing_list(i[0]) for i in configuration_list]
            for img in darknet_images:
                darknet.free_image(img)
            darknet_images = [darknet.make_image(i[1], i[1], 3) for i in configuration_list]
            # for det in detections:
            #     darknet.free_detections(det[1], len(det[1]))
            detections.clear()
            logger.debug('{} got new configuration list {}'.format(thread_name, configuration_list))
        elif msg[0] == 'stop':
            logger.info('Thread {} is killed.'.format(thread_name))
            det_queue.put('stop')
            return 0
        elif msg[0] == 'frame':
            total_detection_time = 0
            frame = msg[1]
            original_width,original_height,_ = frame.shape
            # groundtruth
            default_frame_resized = cv2.resize(
                frame, (default_resolution, default_resolution),interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(default_img, default_frame_resized.tobytes())

            prev_time = time.time()
            default_dets = darknet.detect_image(network, class_names, default_img, thresh=args.thresh)
            default_detection_time = time.time()-prev_time

            default_dets = resizebbox(default_dets, default_resolution, default_resolution, original_width, original_height)
            det_queue.put(default_dets)

            for i in range(len(configuration_list)):
                frame_list = frame_lists[i]
                darknet_image = darknet_images[i]
                resolution = configuration_list[i][1]
                if frame_list.count(cur_frame) != 0:
                    for det_idx in range(len(detections)):
                        if detections[det_idx][0] == configuration_list[i]:
                            # darknet.free_detections(detections[det_idx][1], len(detections[det_idx][1]))
                            detections.pop(det_idx)
                            break

                    frame_resized = cv2.resize(frame, (resolution, resolution),interpolation=cv2.INTER_LINEAR)
                    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

                    prev_time = time.time()
                    detection = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
                    detection_time = time.time()-prev_time

                    detection = resizebbox(detection, resolution, resolution, original_width, original_height)
                    detections.append([configuration_list[i],detection,detection_time])

                    total_detection_time = total_detection_time+detection_time
                    logger.debug(
                        '{}-Second {}, Frame {}, Config {} processed. Detection time {}.'.\
                            format(thread_name, cur_t, cur_frame, configuration_list[i], detection_time)
                    )
                else:
                    for det_idx in range(len(detections)):
                        if detections[det_idx][0] == configuration_list[i]:
                            detections[det_idx][2] = 0
                            break

            det_queue.put(detections.copy())
            det_queue.put([default_detection_time,total_detection_time])
            cur_t, cur_frame = update_time(cur_t, cur_frame)
            control_queue.put(' ')

    logger.info('{}-----Processing ended.'.format(thread_name))
    for img in darknet_images:
        darknet.free_image(img)
    darknet.free_network_ptr(network)
    return 0

def frame_processing_list(frame_rate):
    frame_list = []
    interval = 30/frame_rate
    cur_fram_no = 0
    while cur_fram_no<30:
        frame_list.append(cur_fram_no)
        cur_fram_no = cur_fram_no+interval
    return frame_list

def load_stream(config_file:str):
    # 文件的每一行是一个视频流文件的地址
    f = open(config_file, 'r')

    # list的每个元素是一个三元组，
    # 元组第一个元素是cv的capture，用于主线程从视频文件获取帧
    # 元组第二个元素是消息队列，用于主线程控制子线程执行
    # 元组第三个元素是处理结果队列，用于在获取了处理结果之后对添加相应的处理
    # 元组第四个元素是线程对ß
    video_processor_list = [] 
    det_queues = []
    
    index = 0
    line = f.readline().strip('\n')
    while line:
        stream_info = line.split(' ')
        filepath = stream_info[0]
        classname = stream_info[1]
        cap = cv2.VideoCapture(filepath)

        network, class_names, _ = darknet.load_network(
            'yolov3.cfg',
            'coco.data',
            'yolov3.weights',
            batch_size=1
        )

        msg_queue = Queue(maxsize=30+1)
        det_queue = Queue(maxsize=30)
        thread = threading.Thread(target=video_processing, args=(cap, msg_queue, det_queue, network, class_names))
        thread.setName('{}'.format(index))
        thread.start()
        logger.info('Video {} loaded.'.format(line))
        video_processor_list.append((cap, msg_queue, det_queue, thread, classname))
        det_queues.append(det_queue)
        index = index + 1
        line = f.readline().strip('\n')
    f.close()

    return video_processor_list, det_queues

def capture_stream(stream_list,cur_t,cur_frame):
    for t in stream_list:
        cap = t[0]
        if cap.isOpened():
            ret, frame = cap.read()
            msg_queue = t[1]
            if not ret:
                cap.release()
                logger.info('Video {} end. Total {} seconds, {} frames.'.format(t[3].getName(),cur_t,cur_frame))
                msg_queue.put(('stop',1))
                continue
            logger.debug('Put frame {} of second {} to queue of Thread {}. Queue full?{}'.\
                format(cur_frame,cur_t,t[3].getName(),t[1].full()))
            msg_queue.put(item=('frame', frame),block=True)
            control_queue.get()

def update_time(time, frame):
    if frame < 29:
        return time, frame+1
    else:
        return time+1, 0

def wait_for_detections():
    # 为了统计处理的准确率，用于评估结果，返回每个流在这一帧的处理的准确率
    f1_score = []
    detection_times = []
    for processor in video_processor_list:
        if processor[2].empty() and not processor[3].is_alive():
            f1_score.append(-1)
            continue
        default_dets = processor[2].get()
        if type(default_dets)==str and default_dets == 'stop':
            logger.info('Det Queue of {} recv stop signal'.format(processor[3].getName()))
            f1_score.append(-1)
            continue
        real_detections = processor[2].get()
        detection_time = processor[2].get()
        # print(default_dets)
        # print(real_detections)
        f1_score.append(f1_evaluate(default_dets, real_detections, processor[4]))
        detection_times.append(detection_time)
    return f1_score,detection_times

def iou(gt, det):
    # print(gt)
    left_gt, top_gt, right_gt, bottom_gt = gt[2][0], gt[2][1], gt[2][0]+gt[2][2], gt[2][1]+gt[2][3]
    left_det, top_det, right_det, bottom_det = det[2][0], det[2][1], det[2][0]+det[2][2], det[2][1]+det[2][3]
    # print(left_gt, top_gt, right_gt, bottom_gt)
    # print(left_det, top_det, right_det, bottom_det)
    if left_det > right_gt:
        return 0
    elif left_gt > right_det:
        return 0
    elif top_gt > bottom_det:
        return 0
    elif top_det > bottom_gt:
        return 0
    else:
        left_inter = max(left_det,left_gt)
        top_inter = max(top_det,top_gt)
        right_inter = min(right_det, right_gt)
        bottom_inter = min(bottom_det,bottom_gt)
        intersection_area = (right_inter-left_inter)*(bottom_inter-top_inter)
        union_area = \
            (right_det-left_det)*(bottom_det-top_det)+(right_gt-left_gt)*(bottom_gt-top_gt)-intersection_area
        iou = np.divide(intersection_area, union_area)
        # print(iou)
        assert iou >= 0
        return iou

def f1_evaluate(gts, dets_list:List[List],classname):
    # dets_list的每个元素是一个配置下的检测结果
    # print(gts)
    # print(dets_list)
    # classname = args.classname
    score = []
    for dets_info in dets_list:
        config = dets_info[0]
        dets = dets_info[1]
        # TODO:每个dets与gts计算f1分数
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for gt in gts:
            flag = False
            max_iou = -1
            argmaxiou = -1
            for i in range(len(dets)):
                iou_tmp = iou(gt, dets[i])
                if iou_tmp>max_iou:
                    max_iou = iou_tmp
                    argmaxiou = i
            if argmaxiou>=0:
                det = dets[argmaxiou]
                if max_iou>iou_threshold:
                    flag = True
                    if det[0] == gt[0] and gt[0] == classname:
                        tp = tp+1
                    elif det[0] == classname and gt[0] != classname:
                        fp = fp+1
                    elif det[0] != classname and gt[0] == classname:
                        fn = fn+1
                    else:
                        tn = tn+1
            if flag == False and gt[0] == classname:
                fn = fn+1
        if len(gts) == 0:
            for det in dets:
                if det[0] == classname:
                    fp = fp+1
        if tp+fp == 0 and fn>0:# 表明det里面没有车，但是可能有车没检测到
            f1 = 0 #recall是0
        elif tp+fn == 0 and fp>0:# 表明gt里面没有车，但是det检测到车了
            f1 = 0 #precision是0
        elif tp+fp+fn==0:# 表明gt和det里都没有车
            f1 = 1
        else:
            precision = np.divide(tp,tp+fp)
            recall = np.divide(tp,tp+fn)
            if precision == 0 or recall == 0:
                f1 = 0
            else:
                f1 = np.divide(2*precision*recall,precision+recall)
        # print('{},{}:{} {} {} {}'.format(config, dets, tp, fn, fp, tn))
        score.append([config,f1,dets_info[2]])
    # print(score)
    return score

def get_data(profiling, profiling_exe, exe, f1_score):
    '''
    参数一：侧写阶段的侧写配置
    参数二：侧写阶段的执行配置
    参数三：执行阶段的执行配置，执行阶段没有侧写过程，没有侧写配置
    参数四：这一阶段中每个配置的准确率，占用的GPU时间以及处理的帧数
    处理过程：
    1.处理得到这一段的准确率以及占用的GPU时间
    2.处理得到这一段的侧写占用的时间
    '''
    cost_and_accuracy = []
    for i in range(num_of_flows):
        profile_acc = 0
        exe_acc = 0
        profiling_cost = 0
        inference_cost = 0
        final_accuracy = 0 
        profiling_frame = profiling_t*30
        exe_frame = (w-profiling_t)*30
        total_time = 0
        data = f1_score[i]
        # print(data)
        for config in data:
            #这个配置是执行配置
            total_time = total_time+config[2]
            if config[0] == profiling_exe[i] and config[0] == exe[i]:
                final_accuracy = config[1]*w*30
                inference_cost = config[2]
                profile_acc = exe_acc = config[1]
            elif config[0] == profiling_exe[i]:
                final_accuracy = config[1]*profiling_frame + final_accuracy
                inference_cost = config[2] + inference_cost
                profile_acc = config[1]
            elif config[0] == exe[i]:
                final_accuracy = config[1]*exe_frame + final_accuracy
                inference_cost = config[2] + inference_cost
                exe_acc = config[1]
        final_accuracy = np.divide(final_accuracy,w*30)
        if profiling:
            for config in data:
                for pf_config in profiling[i]:
                    # 这个配置是侧写配置
                    if pf_config == config[0]:
                        # 这里要分情况讨论，可能这个配置同时也是执行配置，就会使开销虚大
                        if pf_config == exe[i]:
                            profiling_cost = profiling_cost + config[2]*np.divide(profiling_frame,w*30)
                        elif pf_config != profiling_exe[i]:
                            profiling_cost = profiling_cost + config[2]
        cost_and_accuracy.append((profiling_cost, inference_cost, total_time, final_accuracy))
        # cost_and_accuracy.append((profiling_cost, inference_cost, total_time, final_accuracy, {profiling_exe[i]:profile_acc}, {exe[i]:exe_acc}))
    return cost_and_accuracy

if __name__ == '__main__':
    args = parse_args()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    # 输出到控制台，不设置format
    handler_stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stream)
    handler_file = logging.FileHandler(args.output, 'w')
    handler_file.setLevel(level=logging.WARN) # 更改level
    formatter = logging.Formatter('%(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)


    video_processor_list, det_queues = load_stream(args.input)
    profiling_t, w, max_segments, iou_threshold, num_of_flows = initialize()

    logger.info('Loading end.')
    
    logger.info('Main controller started.')
    cur_segment = 0

    ProcessingEnded = False
    while cur_segment < max_segments and not ProcessingEnded:
        f1_score_cache = [[] for i in range(num_of_flows)]
        cur_t, cur_frame = update_time(0, -1)
        total_t = cur_segment*w

        while cur_t < w and not ProcessingEnded:
            # subsegment的调度
            schedule_list = scheduler.MYSolution.schedule(cur_segment, cur_t, cur_frame)
            # schedule_list = scheduler.PeriodicMethod.schedule(cur_segment, cur_t, cur_frame)
            # schedule_list = scheduler.OneTimeMethod.schedule(cur_segment, cur_t, cur_frame)
            if schedule_list:
                logger.info('Schedule for segment {}:{}'.format(cur_segment,schedule_list))
                for i in range(num_of_flows):
                    mq = video_processor_list[i][1]
                    mq.put(('configuration_list',schedule_list[i].copy()))
                schedule_list.clear()
            capture_stream(video_processor_list, total_t+cur_t, cur_frame)

            f1_score, detection_times = wait_for_detections()
            logger.debug('f1 score of second {} frame {}:{}.'.format(total_t+cur_t, cur_frame,f1_score))
            for i in range(num_of_flows):
                if type(f1_score[i]) == list:
                    for f1 in f1_score[i]:
                        c_flag = False
                        for c in f1_score_cache[i]:
                            if c[0] == f1[0]:
                                c[1] = c[1]+f1[1]
                                c[2] = c[2]+f1[2]
                                c[3] = c[3]+1
                                c_flag = True
                        if not c_flag:
                            f1_tmp = f1.copy()
                            f1_tmp.append(1)
                            f1_score_cache[i].append(f1_tmp)
            # 收集信息
            scheduler.MYSolution.update_valuation(cur_segment, cur_t, cur_frame, f1_score)
            # scheduler.PeriodicMethod.update_valuation(cur_segment, cur_t, cur_frame, f1_score)
            # scheduler.OneTimeMethod.update_valuation(cur_segment, cur_t, cur_frame, f1_score)

            ProcessingEnded = True
            for processor in video_processor_list:
                if processor[0].isOpened():
                    ProcessingEnded = False
            tmp_second = cur_t
            cur_t, cur_frame = update_time(cur_t, cur_frame)
            if tmp_second < cur_t:
                logger.info('Frame of second {} processed.'.format(tmp_second))

        for i in range(num_of_flows):
            for f1 in f1_score_cache[i]:
                f1[1] = np.divide(f1[1], f1[3])
        logger.debug('f1 score of segment {}:\n{}'.format(cur_segment, f1_score_cache))
        profiling, profilingexe, exe=scheduler.MYSolution.get_schedule()
        # profiling, profilingexe, exe=scheduler.PeriodicMethod.get_schedule()
        # profiling, profilingexe, exe=scheduler.OneTimeMethod.get_schedule()
        logger.info('Schedule:\n{}\n{}\n{}'.format(profiling,profilingexe,exe))
        d = get_data(profiling,profilingexe,exe,f1_score_cache)
        logger.warning('Segment {} performance:{}'.format(cur_segment, d))
        cur_segment = cur_segment + 1

    if not ProcessingEnded:
        for processor in video_processor_list:
            logger.info('Try to kill thread {}.'.format(processor[3].getName()))
            processor[1].put(('stop',1))
    # print(ProcessingEnded)
    for processor in video_processor_list:
        processor[3].join()
        processor[0].release()
        logger.info('Thread {} end.'.format(processor[3].getName()))
    logger.info('Video analysis end.')
