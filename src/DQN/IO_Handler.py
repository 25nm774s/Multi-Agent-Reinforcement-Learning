import torch

class Model_IO:
    def save(self, file_path, state_dict): 
        torch.save(state_dict, file_path)

    def load(self, file_path) -> dict:
        try: 
            return torch.load(file_path, weights_only=True)
        except Exception as e: 
            raise ValueError(f"Error loading model: {e}")

    def save_checkpoint(self, file_path, model_state, target_state, optimizer_state, epoch, epsilon, beta=None): 
        torch.save({'model_state': model_state, 'target_state': target_state, 'optimizer_state': optimizer_state, 'epoch': epoch, 'epsilon': epsilon, 'beta': beta}, file_path)

    def load_checkpoint(self, file_path) -> dict:
        try: 
            return torch.load(file_path, weights_only=False)
        except Exception as e: 
            raise ValueError(f"Error loading checkpoint: {e}")
