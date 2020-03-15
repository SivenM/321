import torch
# Создание тензоров
# Создаем тензор из нулей, где 3 = строки, 4 = столбцы
x = torch.zeros([3, 4])
#print(x)
# Создаем тензор из единиц, где 3е число 4ка является количеством плоскотей
y = torch.ones(3, 4, 4)
#print(y)

z = torch.Tensor([[1,  2,  3,  4],
                  [5,  6,  7,  8],
                  [9, 10, 11, 12]])
print(z)
print(z.shape)

# Градиент
a = torch.tensor(
    [[1.,  2.,  3.,  4.],
     [5.,  6.,  7.,  8.],
     [9., 10., 11., 12.]])

#######
device = torch.device('cuda:0' 
                      if torch.cuda.is_available() 
                      else 'cpu')
a = a.to(device)
#######
a.requires_grad_() #из константы делаем переменную, по кторой будем считать производную

function = 10 * (a ** 2).sum()

function.backward() # backward() считает производные фукции

print(a.grad, '<- gradient')

alpha = 0.01
a.data -= alpha * a.grad
a.grad.zero_()

"asdasdadadadadasdadasdadadasd"