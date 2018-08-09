import  torchvision
import  torch
from torch.autograd import Variable
a =Variable(torch.randn(5, 3, 224, 224))
torch.split(a,5,0)
print(a[0])
vgg16featuremap = torchvision.models.vgg16(pretrained=True).features
conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-8])
y = conv1_conv4(a)
y_l2norm = torch.norm(y, 2, 1)
y_mean = torch.mean(torch.mean(y_l2norm, 1),1)
y_max =torch.max(torch.max(y_l2norm, 1)[0],1)[0]

n1 = y_l2norm.size()[1]
n2 = y_l2norm.size()[2]
a = []
b = []
for i in n1 :
    for j in n2:
        if  y_l2norm[:,i,j] > y_mean[:]
            torch.max(a)
