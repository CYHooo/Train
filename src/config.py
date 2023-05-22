from src.utils.hyperparameter import hyperparameter

class Config():
    def __init__(self,txt_file) -> None:
        self.txt_file = txt_file

    def hyperparameter(self):    
        task, model, dir_data, dir_label, epochs, lr, batch_size, optimizer = hyperparameter(self.txt_file)
        cfg = {
            "task":task, 
            "model":model, 
            "dir_data":dir_data, 
            "dir_label":dir_label, 
            "epochs":epochs, 
            "lr":lr, 
            "batch_size":batch_size, 
            "optimizer":optimizer
        }
        
        print('-'*40)
        print('CONFIG:')
        print('-'*40)
        print(f"task: {task}\nmodel: {model} \
              \ndir_data: {dir_data}\
              \ndir_label: {dir_label}\
              \nepoch: {epochs}\
              \nlearning_rate: {lr}\
              \nbatch_size: {batch_size}\
              \noptimizer: {optimizer}")
        print('-'*40)
        
        return cfg