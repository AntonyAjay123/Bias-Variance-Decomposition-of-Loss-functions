#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample


# In[6]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import mode


# In[5]:


def train_model(train_data_list,loss_fn,lr,model_class,model_kwargs,num_models,X_test,max_epochs,save_path,device,batch_size,patience=10,task='regression'):
    all_preds=[]
    for i, train_data in enumerate(train_data_list):
        print(f'\n--- Training Model {i+1}/{num_models} ---')

        X_train_resampled = train_data[0]
        y_train_resampled = train_data[1]

        # Split the resampled data into training and validation sets for early stopping
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_resampled, y_train_resampled, test_size=0.2, random_state=42
        )

        ## Creating tensors for the train and validation data
        X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32).to(device)
        if task=='classification':
            y_train_tensor = torch.tensor(y_train_split, dtype=torch.long).to(device)
            y_val_tensor = torch.tensor(y_val_split, dtype=torch.long).to(device)
        elif task == 'regression':
            y_train_tensor = torch.tensor(y_train_split, dtype=torch.float32).view(-1, 1).to(device)
            y_val_tensor = torch.tensor(y_val_split, dtype=torch.float32).view(-1, 1).to(device)
            
        
        X_val_tensor = torch.tensor(X_val_split, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        ## Data Loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        ## Model initialization
        model = model_class(**model_kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0

        # Model Training loop with early stopping
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = loss_fn(val_output, y_val_tensor)
            
            print(f'Epoch {epoch+1}/{max_epochs}, Avg Train Loss: {epoch_loss/len(train_dataset):.4f}, Val Loss: {val_loss.item():.4f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt') # Save the best model
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load the best model to use for prediction
        model.load_state_dict(torch.load('best_model.pt'))

        ## Model evaluation
        model.eval()
        if task == 'classification':
            with torch.no_grad():
                logits = model(X_test_tensor)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
        elif task == 'regression':
            model.eval()
            with torch.no_grad():
                preds = model(X_test_tensor).cpu().numpy()
                all_preds.append(preds)
        
    return all_preds

# In[3]:


def get_bvd_mse(all_preds_np,y_test):
    mean_preds = np.mean(all_preds_np, axis=0)  # shape: (num_test_samples,)
    bias = np.mean((mean_preds - y_test.squeeze()) ** 2)
    # variance = np.mean(np.var(all_preds_np, axis=0))
    variance = np.mean((all_preds_np - mean_preds[None, :]) ** 2) 
    return bias,variance


# In[ ]:


def get_bvd_mae(model,test_loader, all_preds):

    y_m = all_preds.median(dim=0).values
    y_o = torch.cat([yb for _, yb in test_loader], dim=0).squeeze() 
    y_o = y_o.unsqueeze(0)

    B=torch.abs(y_o-y_m)
    V = torch.mean(torch.abs(all_preds-y_m.unsqueeze(0)),dim=0)
    # Bias effect sign condition
    sbias = torch.ones_like(all_preds)
    sbias = torch.where(
    ((y_o > all_preds) & (y_o < y_m)) |
    ((y_o < all_preds) & (y_o > y_m)),
    -1, 1)
    # Bias effect: (P(s = 1) - P(s = -1)) * B
    p_pos = (sbias == 1).float().mean(dim=0)  # Shape: [N]
    p_neg = (sbias == -1).float().mean(dim=0)  # Shape: [N]
    bias_effect = (p_pos - p_neg) * B  # Shape: [N]
    bias_effect_final = bias_effect.mean().item()

    # Step 6: Variance magnitude
    abs_dev = (all_preds - y_m.unsqueeze(0)).abs()  # Shape: [M, N]

    # Variance effect sign condition
    svar = torch.where(
    ((y_o > all_preds) & (all_preds > y_m.unsqueeze(0))) |
    ((y_o < all_preds) & (all_preds < y_m.unsqueeze(0))),
    -1, 1)
    neg_mask = (svar == -1).float()
    P_neg = neg_mask.mean(dim=0)  # [N]
    neg_contrib = (abs_dev * neg_mask).sum(dim=0) / (neg_mask.sum(dim=0) + 1e-8) # [N]

    V = abs_dev.mean(dim=0)  # [N]
    variance_effect = V - 2 * neg_contrib * P_neg  # [N]
    variance_effect_final = variance_effect.mean().item()

    return bias_effect_final, variance_effect_final


# In[ ]:


def estimate_bias_variance(model_class, X_train, y_train, X_test, y_test, loss_fn, model_kwargs={},
                           num_models=20, max_epochs=100, patience=10, batch_size=64, lr=0.001, device='cpu'):

    # Store predictions from each model on the test set
    all_preds = []
    
    # Create num_models bootstrapped training sets
    train_data_list = [resample(X_train, y_train, replace=True) for _ in range(num_models)]

    print(f"Starting experiment with {num_models} models...")

    for i, train_data in enumerate(train_data_list):
        print(f'\n--- Training Model {i+1}/{num_models} ---')

        X_train_resampled = train_data[0]
        y_train_resampled = train_data[1]

        # Split the resampled data into training and validation sets for early stopping
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_resampled, y_train_resampled, test_size=0.2, random_state=42
        )

        ## Creating tensors for the train and validation data
        X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_split, dtype=torch.float32).view(-1, 1).to(device)
        X_val_tensor = torch.tensor(X_val_split, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_split, dtype=torch.float32).view(-1, 1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        ## Data Loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        ## Model initialization
        model = model_class(**model_kwargs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0

        # Model Training loop with early stopping
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = loss_fn(val_output, y_val_tensor)
            
            print(f'Epoch {epoch+1}/{max_epochs}, Avg Train Loss: {epoch_loss/len(train_dataset):.4f}, Val Loss: {val_loss.item():.4f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt') # Save the best model
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load the best model to use for prediction
        model.load_state_dict(torch.load('best_model.pt'))

        ## Model evaluation
        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor).cpu().numpy()
            all_preds.append(preds)
    
    # Continue with your existing calculations
    all_preds_np = np.stack(all_preds, axis=0).squeeze()
    y_test = y_test.reshape(-1, 1)

    # Bias-Variance Decomposition
    if isinstance(loss_fn, nn.MSELoss):
        bias, variance = get_bvd_mse(all_preds_np, y_test)
    elif isinstance(loss_fn, nn.L1Loss):
        all_preds_tensor = torch.tensor(all_preds, dtype=torch.float32)
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        bias, variance = get_bvd_mae(model, test_loader, all_preds_tensor)
    else:
        raise ValueError("Unsupported loss function")

    # Expected MSE (avg total error)
    if isinstance(loss_fn, nn.MSELoss):
        total_error = np.mean((all_preds_np - y_test.squeeze()) ** 2)
    elif isinstance(loss_fn, nn.L1Loss):
        y_test = y_test.squeeze()
        total_error = np.mean(np.abs(np.median(all_preds_np, axis=0) - y_test))

    error_sum = bias + variance

    print(f"\n--- Final Results ---")
    print(f"Bias²:    {bias:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Total error: {total_error:.4f}")
    print(f"Bias² + Variance: {(bias + variance):.4f}")
    
    return bias, variance, total_error, error_sum


# In[ ]:


def get_bias_variance_0_1(model_class,loss_fn, X_train, y_train, X_test, y_test, model_kwargs={},
                              num_models=20, max_epochs=100, patience=10,
                              batch_size=64, lr=0.001, device='cpu', save_path='best_model.pt'):
    train_data_list= [resample(X_train,y_train,replace=True) for _ in range(num_models)]
    all_preds = train_model(train_data_list,loss_fn,lr,model_class,model_kwargs,num_models,X_test,max_epochs,save_path,device,batch_size,patience,task='classification')

    # Convert to (num_runs, num_test_samples)
    all_preds = np.stack(all_preds, axis=0)
    print(f"all predictions {all_preds}")
    y_test = y_test.flatten()

    ## predicting the mode
    mode_preds, _ = mode(all_preds,axis=0,keepdims=False)
    print(f"mode preds {mode_preds}")
    mode_preds = np.array(mode_preds).flatten()
    y_test = np.array(y_test).flatten()

    bias_arr = (mode_preds!=y_test).astype(float)

    var_arr = (all_preds!=mode_preds).mean(axis=0)

    expected_loss_arr = np.where(bias_arr == 0,var_arr,1 - var_arr)
    avg_bias = bias_arr.mean()
    avg_var = var_arr.mean()
    avg_exp_loss_paper = expected_loss_arr.mean()
    avg_exp_loss = (all_preds != y_test).mean()
    empirical_01_loss = (all_preds!=y_test).mean()
    print(f"\n--- Final 0-1 Loss Decomposition Results ---")
    print(f"Average Bias         : {avg_bias:.4f}")
    print(f"Average Variance     : {avg_var:.4f}")
    print(f"Expected 0-1 Loss    : {avg_exp_loss:.4f}")
    print(f"Expected 0-1 Loss paper   : {avg_exp_loss_paper:.4f}")
    print(f"Empirical 0-1 Loss   : {empirical_01_loss:.4f}")
    print(f"Bias - Variance?     : {avg_bias - avg_var:.4f} ")

    return avg_bias, avg_var, avg_exp_loss, empirical_01_loss, {
        'bias': bias_arr,
        'variance': var_arr,
        'expected_loss': expected_loss_arr
    }
# train_model(train_loader,train_dataset,loss_fn,lr,model,X_val_tensor,y_val_tensor,max_epochs,save_path,patience=10)

# In[ ]:





# In[ ]:




