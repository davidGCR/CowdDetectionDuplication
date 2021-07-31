import os
import sys
# g_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(g_path)
# sys.path.insert(1, g_path)

import argparse
import cv2
import torch
import numpy as np
import re
import json

sys.path.insert(0, '../lib')
from utils import misc_utils, visual_utils, nms_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(args, config, network):
    saveDir = os.path.join('../model', args.model_dir, config.model_dir)
    model_file = os.path.join(saveDir,'{}'.format(args.resume_weights))
    print('model_file: ', model_file)
    assert os.path.exists(model_file)
    # build network
    net = network().to(device)
    net.eval()
    if device == torch.device("cuda"):
        check_point = torch.load(model_file)
    else:
        check_point = torch.load(model_file, map_location=torch.device('cpu'))
    net.load_state_dict(check_point['state_dict'])
    return net


def inference(img_path, config, net, save=True):
    # model_path
    misc_utils.ensure_dir('outputs')
    
    # get data
    image, resized_img, im_info = get_data(img_path, config.eval_image_short_size, config.eval_image_max_size) 
    pred_boxes = net(resized_img, im_info)

    pred_boxes = pred_boxes.cpu().numpy()
    im_info = im_info.cpu().numpy()

    # print("im_info: ", type(im_info), im_info.size())

    pred_boxes = post_process(pred_boxes, config, im_info[0, 2])
    pred_tags = pred_boxes[:, 5].astype(np.int32).flatten()
    pred_tags_name = np.array(config.class_names)[pred_tags]
    return image, pred_boxes, pred_tags_name

def post_process(pred_boxes, config, scale):
    if config.test_nms_method == 'set_nms':
        assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        top_k = pred_boxes.shape[-1] // 6
        n = pred_boxes.shape[0]
        pred_boxes = pred_boxes.reshape(-1, 6)
        idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
        pred_boxes = np.hstack((pred_boxes, idents))
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.set_cpu_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep]
    elif config.test_nms_method == 'normal_nms':
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.cpu_nms(pred_boxes, config.test_nms)
        pred_boxes = pred_boxes[keep]
    elif config.test_nms_method == 'none':
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
    #if pred_boxes.shape[0] > config.detection_per_image and \
    #    config.test_nms_method != 'none':
    #    order = np.argsort(-pred_boxes[:, 4])
    #    order = order[:config.detection_per_image]
    #    pred_boxes = pred_boxes[order]
    # recovery the scale
    pred_boxes[:, :4] /= scale
    keep = pred_boxes[:, 4] > config.visulize_threshold
    pred_boxes = pred_boxes[keep]
    return pred_boxes

def get_data(img_path, short_size, max_size):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # print("image original : ", img_path)
    resized_img, scale = resize_img(
            image, short_size, max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    resized_img = resized_img.transpose(2, 0, 1) #800x800

    # print("image resize shape: ", resized_img.shape)

    im_info = np.array([height, width, scale, original_height, original_width, 0])
    return image, torch.tensor([resized_img]).float().to(device), torch.tensor([im_info]).to(device)

def resize_img(image, short_size, max_size):
    height = image.shape[0]
    width = image.shape[1]
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    resized_image = cv2.resize(
            image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale


from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def videoDetections_2_JSON(output_path: str, video_detections: list):
    with open(output_path, 'w') as fout:
        json.dump(video_detections , fout, cls=NumpyArrayEncoder)

def JSON_2_videoDetections(json_file):
    with open(json_file, "r") as read_file:
        decodedArray = json.load(read_file)
        # print("decoded Array:", type(decodedArray), len(decodedArray))
        
        for f in decodedArray:
            f['pred_boxes'] = np.asarray(f['pred_boxes'])
        # print(decodedArray[0])
        return decodedArray

def plot_image_detections(decodedArray, dataset_path, save_path=None):
    for item in decodedArray:
        img_path = os.path.join(dataset_path, item['split'], item['video'], item['fname'])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        pred_boxes = item['pred_boxes']
        pred_tags_name = item['tags']
        print(item)
        print('iamge_size: ', image.shape) 
        if pred_boxes.shape[0] != 0:
            
            image = visual_utils.draw_boxes(image,
                                            pred_boxes[:, :4],
                                            scores=pred_boxes[:, 4],
                                            tags=pred_tags_name,
                                            line_thick=1, 
                                            line_color='white')
        name = img_path.split('/')[-1].split('.')[-2]

        if save_path:
            fpath = '{}/{}.png'.format(save_path, name)
            cv2.imwrite(fpath, image)
        cv2.imshow(name, image)
        key = cv2.waitKey(500)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break


def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default="rcnn_emd_refine", type=str)
    parser.add_argument('--resume_weights', '-r', default="rcnn_emd_refine_mge.pth", type=str)
    # parser.add_argument('--img_path', '-i', default="/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames/train/Fight/_6-B11R9FJM_0/frame79.jpg", type=str)
    parser.add_argument('--save_path', '-sp', default="", type=str)
    args = parser.parse_args()
    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    # model_root_dir = os.path.join(g_path,'model', args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config import config
    from network import Network
    # inference(args, config, Network)
    config.eval_image_short_size = 224
    config.eval_image_max_size = 224
    net = build_model(args, config, Network)

    # dataset_dir = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames"
    # dataset_dir = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/HockeyFightsDATASET/frames"
    # dataset_dir = "/media/david/datos/Violence DATA/RWF-2000/frames"
    # dataset_dir = "/content/DATASETS/RWF-2000/frames"
    
    dataset_dir = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips"

    # splits = ["train/Fight", "train/NonFight", "val/Fight", "val/NonFight"]
    # splits = ["violence", "nonviolence"]
    splits = ["anomaly"]

    # folder_out = os.path.join("outputs", "rwf")
    # folder_out = os.path.join("/content/drive/MyDrive/VIOLENCE DATA/PersonDetections", "RWF-2000")
    # folder_out = os.path.join("/media/david/datos/Violence DATA/PersonDetections", "RWF-2000-224")
    # folder_out = os.path.join("/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections", "hockey")
    folder_out = os.path.join("/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections", "ucfcrime2local")
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
        for s in splits:
            os.makedirs(os.path.join(folder_out,s))

    for sp in splits:
        videos_list = os.listdir(os.path.join(dataset_dir, sp)) #list of videos folders
        # one_video = videos_list[0]
        for j, one_video in enumerate(videos_list):
            print("Video {}: {}/{}".format(sp+'/'+one_video, j+1, len(videos_list)))
            video_frames = [frame for frame in os.listdir(os.path.join(dataset_dir, sp, one_video)) if ".jpg" in frame]
            video_frames.sort(key=lambda f: int(re.sub('\D', '', f)))
            video_frames = [os.path.join(dataset_dir, sp, one_video, frame) for frame in video_frames]
            
            video_detections = []
            for i, img_path in enumerate(video_frames):
                # print("\t processing: {}/{}".format(i+1, len(video_frames)))
                image, pred_boxes, pred_tags_name = inference(img_path, config, net)
                # print("pred_boxes: ", pred_boxes, pred_boxes.shape)
                # print("scores: ", pred_boxes[:, 4], pred_boxes[:, 4].shape)
                # print("pred_tags_name: ", pred_tags_name, pred_tags_name.shape)

                frame_detections = {
                    "fname": os.path.split(img_path)[1],
                    "video": one_video,
                    "split": sp,
                    "pred_boxes":  pred_boxes[:, :5],
                    "tags": pred_tags_name
                }
                video_detections.append(frame_detections)

                # # #make folder
                # out_path = os.path.join("outputs", one_video)
                # if not os.path.isdir(out_path):
                #     os.mkdir(out_path)
                # # inplace draw
                # image = visual_utils.draw_boxes(
                #         image,
                #         pred_boxes[:, :4],
                #         scores=pred_boxes[:, 4],
                #         tags=pred_tags_name,
                #         line_thick=1, line_color='white')
                # name = img_path.split('/')[-1].split('.')[-2]
                # fpath = '{}/{}.png'.format(out_path, name)
                # cv2.imwrite(fpath, image)
            # #make folder
            # out_path = os.path.join("outputs", one_video)
            # if not os.path.isdir(out_path):
            #     os.mkdir(out_path)
            videoDetections_2_JSON(os.path.join(folder_out, sp, one_video+".json"), video_detections)
        

if __name__ == '__main__':
    run_inference()
    # decodedArray = JSON_2_videoDetections("/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local/anomaly/Burglary043(464-502).json")
    # plot_image_detections(decodedArray, 
    #                         "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips",
    #                         None)#"/Users/davidchoqueluqueroman/Documents/CODIGOS_SOURCES/CowdDetectionDuplication/results")

