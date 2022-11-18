import os
from tqdm import tqdm
import onnx, onnxruntime
from PIL import Image
from torchvision import transforms
import torch

num = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = r'D:\yiguohuang\teach_people\people\resNet34.onnx'
onnx_model = onnx.load(model_name)
onnx.checker.check_model(onnx_model)
image_dir = r"D:\yiguohuang\teach_people\people\testdataset\testdataset\test\\"
for class_dir in os.listdir(image_dir):
    class_path = os.path.join(image_dir,class_dir)
    class_num = len(os.listdir(class_path))
    for filename in tqdm(os.listdir(class_path)):
        print(class_dir)
        image_path = os.path.join(class_path,filename)

        image = Image.open(image_path).convert('RGB')
        resize = transforms.Compose(
                        [ transforms.Resize((512,384)), transforms.ToTensor()])
        image = resize(image)
        image = image.unsqueeze(0) # add fake batch dimension
        image = image.to(device)

        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        ort_session = onnxruntime.InferenceSession(model_name, providers=EP_list)

        def to_numpy(tensor):
              return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
        ort_outs = ort_session.run(None, ort_inputs)

        max = float('-inf')
        max_index = -1
        for i in range(0, len(ort_outs[0][0])):
            if(ort_outs[0][0][i] > max):
                max = ort_outs[0][0][i]
                max_index = i
                if max_index == class_dir:
                    num += 1
    acc = num / class_num
    print("{}的准确率是{}".format(class_dir,acc))
    num = 0
    acc = 0
