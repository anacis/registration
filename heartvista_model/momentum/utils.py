import torch 

def fft2c(tensor):
    tensor = torch.fft.ifftshift(tensor, dim=(-2, -1))
    tensor = torch.fft.fft2(tensor)
    return torch.fft.fftshift(tensor, dim=(-2, -1))


def ifft2c(tensor):
    tensor = torch.fft.ifftshift(tensor, dim=(-2, -1))
    tensor = torch.fft.ifft2(tensor)
    return torch.fft.fftshift(tensor, dim=(-2, -1))


def ifft1c(tensor, dim=-1):
    tensor = torch.fft.ifftshift(tensor, dim=dim)
    tensor = torch.fft.ifft(tensor, dim=dim)
    return torch.fft.fftshift(tensor, dim=dim)


def fft1c(tensor, dim=-1):
    tensor = torch.fft.ifftshift(tensor, dim=dim)
    tensor = torch.fft.fft(tensor, dim=dim)
    return torch.fft.fftshift(tensor, dim=dim)