# filename = 'hyperparameter.txt'
def hyperparameter(txt_file):
    # txt파일 읽어오기
    with open(txt_file, 'r', encoding='UTF8') as file:
        for line in file:
            if not line.startswith('#'):
                print(line)
            content = file.read()

    variables = {}
    lines = content.split('\n')

    # txt파일의 내용을 딕셔너리 타입으로 변경
    for line in lines:
        # Split the line at '='
        parts = line.split('=')

        if len(parts) == 2:
    
            variable = parts[0].strip()
            value = parts[1].strip()

            variables[variable] = value

    # txt파일에 포함된 하이퍼파라미터들, 필요시 타입에 맞춰 추가
    Task = int(variables['Task'])
    model = str(variables['model'])
    dir_data = str(variables['dir_data'])
    dir_label = str(variables['dir_label'])
    epochs = int(variables['epoch'])
    lr = float(variables['lr'])
    batch_size = int(variables['batch_size'])
    optimizer = str(variables['optimizer'])
    
    return Task, model, dir_data, dir_label, epochs, lr, batch_size, optimizer


if __name__ == "__main__":
    Task, model, dir_data, dir_label, epochs, lr, batch_size, optimizer = hyperparameter('training_parameter.txt')
    print(Task)
    print(model)
    print(dir_data)
    print(dir_label)
    print(epochs)
    print(lr)
    print(batch_size)
    print(optimizer)