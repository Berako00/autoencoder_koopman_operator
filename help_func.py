import torch
import torch.nn.functional as F
import os

from loss_func import custom_loss

def get_model_path(i, user_dir=None):
    path5 = f"/home/trarity/master/koopman_operator/data/"
    path2 = f"C:/Users/jokin/Desktop/Uni/Aalborg/Master/Masters_Thesis/Path/"
    path3 = f"/content/drive/My Drive/Colab Notebooks/"
    path4 = f"/content/drive/MyDrive/Colab Notebooks/"
    path1 = f"{user_dir}/"
    if os.path.exists(path1):
        path1 = f"{path1}Autoencoder_model_params{i}.pth"
        return path1
    elif os.path.exists(path2):
        path2 = f"{path2}Autoencoder_model_params{i}.pth"
        return path2
    elif os.path.exists(path3):
        path3 = f"{path3}Autoencoder_model_params{i}.pth"
        return path3
    elif os.path.exists(path5):
        path5 = f"{path5}Autoencoder_model_params{i}.pth"
        return path5
    else:
        path4 = f"{path4}Autoencoder_model_params{i}.pth"
        return path4

def set_requires_grad(layers, requires_grad):
    for param in layers:
        param.requires_grad = requires_grad

def self_feeding(model, xuk, Num_meas):
    initial_input = xuk[:, 0, :]
    num_steps = int(len(xuk[0, :, 0]))
    inputs = xuk[:,:,Num_meas:]

    predictions = []
    predictions.append(initial_input)

    for step in range(num_steps - 1):
        x_pred = model(initial_input)
        x_pred = torch.cat((x_pred, inputs[:, step, :]), dim=1)
        predictions.append(x_pred.detach())
        initial_input = x_pred

    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xuk)

    return predictions, loss


def enc_self_feeding(model, xuk, Num_meas):
    x_k = xuk[:, 0, :Num_meas]
    u = xuk[:, :, Num_meas:]

    num_steps = int(len(xuk[0, :, 0]))
    predictions = []
    predictions.append(x_k)

    y_k = model.x_Encoder(x_k)
    for m in range(0, num_steps-1):

        v = model.u_Encoder(torch.cat((x_k, u[:, m, :]), dim=1))
        y_k = model.x_Koopman_op(y_k) + model.u_Koopman_op(v)
        x_k = model.x_Decoder(y_k)
        predictions.append(x_k)

    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xuk[:, :, :Num_meas])

    return predictions, loss

def enc_self_feeding_uf(model, xuk, Num_meas):
    x_k = xuk[:, 0, :Num_meas]

    num_steps = int(len(xuk[0, :, 0]))
    predictions = []
    predictions.append(x_k)

    y_k = model.x_Encoder(x_k)
    for m in range(0, num_steps-1):

        y_k = model.x_Koopman_op(y_k)
        x_k = model.x_Decoder(y_k)
        predictions.append(x_k)

    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xuk[:, :, :Num_meas])

    return predictions, loss
