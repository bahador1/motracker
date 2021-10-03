import cv2 as cv
from motrackers.detectors import TF_SSDMobileNetV2
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks


def main(video_path, model, tracker):
    cap = cv.VideoCapture(video_path)
    while True:
        ok, image = cap.read()

        if not ok:
            print("Cannot read the video feed.")
            break

        image = cv.resize(image, (700, 500))

        bboxes, confidences, class_ids = model.detect(image)
        tracks = tracker.update(bboxes, confidences, class_ids)
        updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)

        updated_image = draw_tracks(updated_image, tracks)

        cv.imshow("image", updated_image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Object detections in input video using TensorFlow model of MobileNetSSD.')

    parser.add_argument(
        '--video', '-v', type=str, default="./../video_data/cars.mp4", help='Input video path.')

    parser.add_argument(
        '--weights', '-w', type=str,
        default="./../pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
        help='path to weights file of tf-MobileNetSSD (`.pb` file).'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default="./../pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
        help='path to config file of Caffe-MobileNetSSD (`.pbtxt` file).'
    )

    parser.add_argument(
        '--labels', '-l', type=str,
        default="./../pretrained_models/tensorflow_weights/ssd_mobilenet_v2_coco_names.json",
        help='path to labels file of coco dataset (`.names` file.)'
    )

    parser.add_argument(
        '--gpu', type=bool, default=False,
        help='Flag to use gpu to run the deep learning model. Default is `False`'
    )

    parser.add_argument(
        '--tracker', type=str, default='IOUTracker',
        help="Tracker used to track objects. Options include ['CentroidTracker', 'CentroidKF_Tracker', "
             "'SORT', IOUTracker]")

    args = parser.parse_args()

    if args.tracker == 'CentroidTracker':
        tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    elif args.tracker == 'CentroidKF_Tracker':
        tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
    elif args.tracker == 'SORT':
        tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    elif args.tracker == 'IOUTracker':
        tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
                             tracker_output_format='mot_challenge')
    else:
        raise NotImplementedError

    model = TF_SSDMobileNetV2(
        weights_path=args.weights,
        configfile_path=args.config,
        labels_path=args.labels,
        confidence_threshold=0.4,
        nms_threshold=0.2,
        draw_bboxes=True,
        use_gpu=args.gpu
    )

    main(args.video, model, tracker)
