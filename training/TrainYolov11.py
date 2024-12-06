from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m-seg.pt")

# Train the model
train_results = model.train(
    data="/home/tori/ws/src/nn_laser_spot_tracking/training/coco_subset_1733499037/data.yaml",  # path to dataset YAML
    epochs=500,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch = 16,
)

# Evaluate model performance on the validation set
# metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()

# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model