import torch
import numpy as np
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    # FeatureExtractor(self.feature_module, target_layers)

    def __init__(self, model, target_layers):
        # FeatureExtractor() 中的model和 ModelOutputs() GradCam() 不同
        # FeatureExtractor() 中的model 代表 ModelOutputs() GradCam() 中的 feature_module 即 model.layer2

        self.model = model
        self.target_layers = target_layers
        self.gradients = []
       # print('target layer:',self.target_layers)

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x) # 在feature_module中不断正传

            if name in self.target_layers: # 如果遇到target_layers 定义一个钩子保存梯度
                #print('saving')
                x.register_hook(self.save_gradient)
                #print('self gradient ',x.grad)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
# grad_cam = GradCam(model=model, feature_module=model.layer1, target_layer_names=["1"], use_cuda=args.use_cuda)
#  GradCam中 self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        #print('feature_module name: ',self.feature_module)
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        feature = []
        for name, module in self.model._modules.items():
            #print('current module name,',name)
            # 在整个model中不断正传 有三种情况
            # 如果遇到feature_module 则跳到FeatureExtractor()中正传，FeatureExtractor()中需要target_layers梯度
            if module == self.feature_module:# 跳入FeatureExtractor()正传，得到整层的参数 和target_layers梯度
                target_activations, x = self.feature_extractor(x)
                feature.append(x)
            elif "wh" in name.lower():# 如果遇到"avgpool"代表即将进入fc，需要压平向量
                wh = module(x)
            elif "hm" in name.lower():  # 如果遇到"avgpool"代表即将进入fc，需要压平向量
                hm = module(x)
                #print('hm shape',hm.shape)
            elif "reg" in name.lower():  # 如果遇到"avgpool"代表即将进入fc，需要压平向量
                reg = module(x)
                #x = x.view(x.size(0),-1)
           # elif "avgpool" in name.lower():
            else:# 其他情况   即在除去self.feature_module的其余三层中正向传播
                x = module(x)
        return target_activations, hm

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)
