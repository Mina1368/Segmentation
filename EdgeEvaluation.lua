-- required libraries
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'image'
require 'torch'

model = torch.load('model-edge.net') -- loading the edge-based segmenation model

local function Rescale(img) -- normalization function
m= img:min()
M = img:max()
img = (img-m)/(M-m)
return img
end

h = 360 --input patch heigth
w = 480 --input patch width
t = 29 -- image file number

img = image.load('image' .. t .. '.png')

i = img:size(2)
j = img:size(3)
k = img:size(1)

-- zeropadding the image

r1 = torch.ceil(i/h);
r2 = torch.ceil(j/w);

input = torch.DoubleTensor(k,r1*h,r2*w)
out = torch.DoubleTensor(r1*h,r2*w)
input[{{},{1,i},{1,j}}] = img


--sliding window over the image

for n = 1,r1*h, h do
for m = 1,r2*w,w do
input1 = input[{{},{n,n+h-1},{m,m+w-1}}]
x = torch.Tensor(1, 3, h, w)
x[{1,{},{},{}}] = input1[{{},{1,h},{1,w}}]
model:evaluate()
x = x:cuda() 
y = model:forward(x)
out1 = torch.DoubleTensor(h,w):copy(torch.squeeze(y[{1,2,{},{}}]))
output1 = Rescale(out1)
out[{{n,n+h-1},{m,m+w-1}}]=output1
end
end
o = out[{{1,i},{1,j}}]
image.save('outputEdge' .. t .. '.png', o)

