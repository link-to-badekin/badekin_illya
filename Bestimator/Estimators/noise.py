import torch

def run_estim(model, dataloader, atack, param, device = None):
  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  change_ans = 0.0
  accur = 0.0
  ASR = 0.0
  model.eval()
  with torch.no_grad():
      for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # Forward
        adv_x = atack(X, device, param)
        outputs = model(X)
        outputs_adv = model(adv_x)
        cur, yp = torch.max(outputs, 1)
        cur_adv, yp_adv = torch.max(outputs_adv, 1)
        change_ans += torch.sum(yp != yp_adv)
        accur += torch.sum(yp_adv == y.data)
        #tem = yp[ ]
        ASR += torch.sum( (yp == y.data) & (yp_adv != y.data) )
  print(f"change_ans = {change_ans/len(dataloader.dataset)}")
  print(f"accur = {accur/len(dataloader.dataset)}")
  print(f" ASR = {ASR/len(dataloader.dataset)}")
  
  return (accur/len(dataloader.dataset), ASR/len(dataloader.dataset ))