import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os
import time
import datetime


from help_func import self_feeding, enc_self_feeding, set_requires_grad, get_model_path, enc_self_feeding_uf, load_model
from loss_func import total_loss, total_loss_forced, total_loss_unforced
from nn_structure import AUTOENCODER

def trainingfcn(eps, check_epoch, lr, batch_size, S_p, T, dt, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder, train_tensor, test_tensor, M, device=None):

  if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  pin_memory = True if device.type == "cuda" else False

  train_dataset = TensorDataset(train_tensor)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
  test_dataset = TensorDataset(test_tensor)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

  Models_loss_list = torch.zeros(M)
  c_m = 0
  
  Model_path = [get_model_path(i) for i in range(M)]
  Running_Losses_Array, Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array = [torch.zeros(M, eps) for _ in range(7)]

  hyperparams = {
        'Num_meas': Num_meas,
        'Num_inputs': Num_inputs,
        'Num_x_Obsv': Num_x_Obsv,
        'Num_x_Neurons': Num_x_Neurons,
        'Num_u_Obsv': Num_u_Obsv,
        'Num_u_Neurons': Num_u_Neurons,
        'Num_hidden_x_encoder': Num_hidden_x_encoder,
        'Num_hidden_u_encoder': Num_hidden_u_encoder,
        'dt': dt
  }
  
  for c_m in range(M):
      model_path_i = Model_path[c_m]
      model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv,
                          Num_x_Neurons, Num_u_Obsv, Num_u_Neurons,
                          Num_hidden_x_encoder, Num_hidden_x_decoder,
                          Num_hidden_u_encoder, Num_hidden_u_decoder).to(device)
      optimizer = optim.Adam(model.parameters(), lr=lr)
      best_test_loss_checkpoint = float('inf')

      running_loss_list, Lgx_list, Lgu_list, L3_list, L4_list, L5_list, L6_list = [torch.zeros(eps) for _ in range(7)]

      start_time = time.perf_counter()

      for e in range(eps):
          model.train()
          running_loss, running_Lgx, running_Lgu, running_L3, running_L4, running_L5, running_L6 = [0.0] * 7

          for (batch_x,) in train_loader:
              batch_x = batch_x.to(device, non_blocking=True)
              optimizer.zero_grad()
              [loss, L_gx, L_gu, L_3, L_4, L_5, L_6] = total_loss(alpha, batch_x, Num_meas, Num_x_Obsv, T, S_p, model)
              loss.backward()
              optimizer.step()
              running_loss += loss.item()
              running_Lgx += L_gx.item()
              running_Lgu += L_gu.item()
              running_L3 += L_3.item()
              running_L4 += L_4.item()
              running_L5 += L_5.item()
              running_L6 += L_6.item()

              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

          running_loss_list[e] = running_loss
          Lgx_list[e] = running_Lgx
          Lgu_list[e] = running_Lgu
          L3_list[e] = running_L3
          L4_list[e] = running_L4
          L5_list[e] = running_L5
          L6_list[e] = running_L6


          # Every 20 epochs, evaluate on the test set and checkpoint if improved.
          if (e + 1) % check_epoch == 0:
              now         = time.perf_counter()
              elapsed_sec = now - start_time
              avg_epoch   = elapsed_sec / (e + 1)
              rem_epochs  = eps - (e + 1)
              eta_sec     = avg_epoch * rem_epochs

              elapsed_str = str(datetime.timedelta(seconds=int(elapsed_sec)))
              eta_str     = str(datetime.timedelta(seconds=int(eta_sec)))

              print(f"Epoch {e+1}/{eps}, "f"Train Loss: {running_loss:.3e}, "f"Elapsed: {elapsed_str}, ETA: {eta_str}")
              model.eval()
              test_running_loss = 0.0
              for (batch_x,) in test_loader:
                  batch_x = batch_x.to(device, non_blocking=True)
                  _, loss = enc_self_feeding(model, batch_x, Num_meas)
                  test_running_loss += loss.item()
              print(f'Checkpoint at Epoch {e+1}: Test Running Loss: {test_running_loss:.3e}')

              # If test loss is lower than the one from the previous checkpoint, save the model.
              if test_running_loss < best_test_loss_checkpoint:
                  best_test_loss_checkpoint = test_running_loss
                  checkpoint = {'state_dict': model.state_dict(), **hyperparams}
                  torch.save(checkpoint, model_path_i)
                  print(f'Checkpoint at Epoch {e+1}: New best test loss, model saved.')

      load_model(model, model_path_i, device)

      Models_loss_list[c_m] = best_test_loss_checkpoint
      Running_Losses_Array[c_m, :] = running_loss_list
      Lgx_Array[c_m, :] = Lgx_list
      Lgu_Array[c_m, :] = Lgu_list
      L3_Array[c_m, :] = L3_list
      L4_Array[c_m, :] = L4_list
      L5_Array[c_m, :] = L5_list
      L6_Array[c_m, :] = L6_list

  # Find the best of the models
  Lowest_loss = Models_loss_list.min().item()

  Lowest_loss_index = int((Models_loss_list == Models_loss_list.min()).nonzero(as_tuple=False)[0].item())
  print(f"The best model has a running loss of {Lowest_loss} and is model nr. {Lowest_loss_index}")

  Best_Model = Model_path[Lowest_loss_index]

  return (Lowest_loss,Models_loss_list, Best_Model, Lowest_loss_index, Running_Losses_Array, Lgx_Array, Lgu_Array, L3_Array, L4_Array, L5_Array, L6_Array)



def trainingfcn_ga(eps, check_epoch, lr, batch_size, S_p, T, dt, alpha, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder, train_tensor, test_tensor, M, device=None):

  if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  pin_memory = True if device.type == "cuda" else False

  train_dataset = TensorDataset(train_tensor)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
  test_dataset = TensorDataset(test_tensor)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

  Models_loss_list = torch.zeros(M)
  c_m = 0
  
  Model_path = [get_model_path(i) for i in range(M)]
  
  for c_m in range(M):
      model_path_i = Model_path[c_m]
      model = AUTOENCODER(Num_meas, Num_inputs, Num_x_Obsv,
                          Num_x_Neurons, Num_u_Obsv, Num_u_Neurons,
                          Num_hidden_x_encoder, Num_hidden_x_decoder,
                          Num_hidden_u_encoder, Num_hidden_u_decoder).to(device)
      optimizer = optim.Adam(model.parameters(), lr=lr)
      best_test_loss_checkpoint = float('inf')

      for e in range(eps):
          model.train()

          for (batch_x,) in train_loader:
              batch_x = batch_x.to(device, non_blocking=True)
              optimizer.zero_grad()
              [loss, L_gx, L_gu, L_3, L_4, L_5, L_6] = total_loss(alpha, batch_x, Num_meas, Num_x_Obsv, T, S_p, model)
              loss.backward()
              optimizer.step()
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

          # Every 20 epochs, evaluate on the test set and checkpoint if improved.
          if (e + 1) % check_epoch == 0:
              model.eval()
              test_running_loss = 0.0
              for (batch_x,) in test_loader:
                  batch_x = batch_x.to(device, non_blocking=True)
                  _, loss = enc_self_feeding(model, batch_x, Num_meas)
                  test_running_loss += loss.item()

              # If test loss is lower than the one from the previous checkpoint, save the model.
              if test_running_loss < best_test_loss_checkpoint:
                  best_test_loss_checkpoint = test_running_loss

      Models_loss_list[c_m] = best_test_loss_checkpoint

  # Find the best of the models
  Lowest_loss = Models_loss_list.min().item()

  Lowest_loss_index = int((Models_loss_list == Models_loss_list.min()).nonzero(as_tuple=False)[0].item())

  Best_Model = Model_path[Lowest_loss_index]

  return (Lowest_loss, Models_loss_list, Best_Model, Lowest_loss_index)
