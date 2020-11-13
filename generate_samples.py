from main import YourModel, generate_samples

hidden_dim = 256
num_samples = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_dir = './model_save' 
restore_path =  "./model_save/model_state_dict.pt"

model = YourModel(hidden_num)

model.load_state_dict(restore_path).to(device)
print('==== Model restored : %s' % restore_path)
model.eval()
generate_samples(model, './model_save/samples', num_samples, hidden_dim)
