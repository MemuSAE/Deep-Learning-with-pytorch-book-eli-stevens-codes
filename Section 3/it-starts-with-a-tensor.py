a = [1.0,2.0,1.0]

a[0]

a[2]=3.0

a

"""##3.2.2:Constructing our First tensors


"""

import torch
a=torch.ones(3)
a

a[1]

float(a[1])

a[2]=2.0
a

"""##3.2.3:Essence of Tensors


"""

points=torch.zeros(6)
points[0]=4.0
points[1]=1.0
points[2]=5.0
points[3]=3.0
points[4]=2.0
points[5]=1.0

points = torch.tensor([4.0,1.0,5.0,3.0,2.0,1.0])
points

float(points[0]),float(points[1])

points = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
points

points.shape

points = torch.zeros(3,2)
points

points = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
points

points[0,1]

points[0]

"""#**3.3 :**

##3.3:Indexing
"""

some_list = list(range(6))
some_list[:]
some_list[1:4]
some_list[1:]
some_list[:4]
some_list[:-1]
some_list[1:4:2]

print(some_list[:],some_list[1:4],some_list[1:],some_list[:4],some_list[:-1],some_list[1:4:2])

points[1:]
points[1:,:]
points[1:,0]
points[None]

print(points[1:],points[1:,:],points[1:,0],points[None])

"""#**3.4 :**

##3.4:Named Tensors
"""

img_t = torch.randn(3,5,5)
weights=torch.tensor([0.2126,0.7152,0.0722])

batch_t = torch.randn(2,3,5,5)

img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
img_gray_naive.shape,batch_gray_naive.shape

unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)
img_weights = (img_t * unsqueezed_weights)
batch_weights = (batch_t * unsqueezed_weights)
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)
batch_weights.shape,batch_t.shape,unsqueezed_weights.shape

img_gray_weighted_fancy = torch.einsum('...chw,c->...hw',img_t,weights)
batch_gray_weighted_fancy = torch.einsum('...chw,c->...hw',batch_t,weights)
batch_gray_weighted_fancy.shape

weights_named = torch.tensor([0.2126,0.7152,0.0722],names=['channels'])
weights_named

img_named=img_t.refine_names(...,'channels','rows','columns')
batch_named=batch_t.refine_names(...,'channels','rows','columns')
print('img named:',img_named.shape,img_named.names)
print('batch named:',batch_named.shape,batch_named.names)

weights_aligned = weights_named.align_as(img_named)
weights_aligned.shape,weights_aligned.names

gray_named = (img_named * weights_aligned).sum('channels')
gray_named.shape,gray_named.names

weights_aligned = weights_named.align_as(img_named)
gray_named = (img_named * weights_aligned).sum('channels')

"""#**3.5 :**

##3.5.3:Managing a Tensor's dtype attribute :
"""

gray_plain = gray_named.rename(None)
gray_plain.shape,gray_plain.names

double_points = torch.ones(10,2,dtype=torch.double)
short_points = torch.tensor([[1,2],[3,4]],dtype=torch.short)
short_points.dtype

double_points = torch.zeros(10,2).double()
short_points = torch.ones(10,2).short()
double_points = torch.zeros(10,2).to(torch.double)
short_points = torch.ones(10,2).to(dtype=torch.short)

points_64=torch.rand(5,dtype=torch.double)
points_short=points_64.to(torch.short)
points_64*points_short

"""#**3.6 :**

##3.6:Tensors of API :
"""

a = torch.ones(3,2)
a_t = torch.transpose(a,0,1)
a.shape,a_t.shape

a=torch.ones(3,2)
a_t = a.transpose(0,1)
a.shape,a_t.shape

"""#**3.7 :**

##3.7.1:Indexing into storage :
"""

points=torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
points.storage()

points_storage = points.storage()
points_storage[0]

points.storage()[1]

points = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
points_storage = points.storage()
points_storage[0] = 2.0
points

"""##3.7.2:Modefying stored values:in-place operations :


"""

a=torch.ones(3,2)
a.zero_()
a

"""#**3.8 :**

##3.8.1:views of another tensor's storage :
"""

points = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
second_point = points[1]
second_point.storage_offset()

second_point.size()

second_point.shape

points.stride()

second_point=points[1]
second_point.size()

second_point.storage_offset()

second_point.stride()

points = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
second_point = points[1]
second_point[0] = 10.0
points

points = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
second_point = points[1].clone()
second_point[0] = 10.0
points

"""##3.8.2:Transposing without copying :


"""

points = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
points

points_t=points.t()
points_t

id(points.storage())==id(points_t.storage())

points.stride()

points_t.stride()

"""##3.8.3:Transposing without copying :


"""

some_t=torch.ones(3,4,5)
transpose_t=some_t.transpose(0,2)
some_t.shape

transpose_t.shape

some_t.stride()

transpose_t.stride()

"""##3.8.4:Contiguous tensors :


"""

points.is_contiguous()

points_t.is_contiguous()

points = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])
points_t=points.t()
points_t

points_t.storage()

points_t.stride()

points_t_cont=points_t.contiguous()
points_t_cont

points_t_cont.stride()

points_t_cont.storage()

"""#**3.9 :**

##3.9.1:Managing tensor's device attribute:
"""

points_gpu = torch.tensor([[4.0,1.0],[5.0,3.0],[2.0,1.0]])

points_gpu = points.to(device='cuda')

points_gpu=points.to(device='cuda:0')

points=2*points
points_gpu=2*points.to(device='cuda')

points_gpu=points_gpu+4

points_cpu=points_gpu.to(device='cpu')

points_gpu=points.cuda()
points_gpu=points.cuda(0)
points_cpu=points_gpu.cpu()

"""#**3.10 :**

##3.10:NumPy interoperability:
"""

points = torch.ones(3,4)
points_np = points.numpy()
points_np

points = torch.from_numpy(points_np)

"""#**3.12 :**

##3.12.1:Managing tensor's device attribute:
"""

!pip install h5py

import h5py
f=h5py.File('/content/ourpoints.hdf5','w')
dset = f.create_dataset('coords',data=points.numpy())
f.close()

f=h5py.File('/content/ourpoints.hdf5','r')
dset=f['coords']
last_points=dset[-2:]

last_points=torch.from_numpy(dset[-2:])
f.close()
