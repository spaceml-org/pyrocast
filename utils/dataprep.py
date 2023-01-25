def load_data(train_list, parent_dir, event_id):
    """ Loads and matches datacubes and flags """
    
    # Convert start, end and timestep to a list of datetimes and a vector of decimal values
    datetime_list, _ = dp.create_daytime_list(train_list)

    # Load files
    path_list, dataset_cubes, flags = dp.load_files2(parent_dir, datetime_list, frequency=10, event_id)
    
    #Get rid of missing and nighttime/twilight data
    dataset_cubes_train = []
    flags_train = []
    datetimes_train = []
    dataset_cubes_night = []
    flags_night = []
    datetimes_night = []
    
    for i in range(len(dataset_cubes)):

        if(flags[i][0]):
            #Set aside nighttime and sunrise/sunset data
            if(flags[i][1]):
                dataset_cubes_train.append(dataset_cubes[i])
                flags_train.append(flags[i][-1])
                datetimes_train.append(datetime_list[i])
            else:
                dataset_cubes_night.append(dataset_cubes[i])
                flags_night.append(flags[i][-1])
                datetimes_night.append(datetime_list[i])

    dataset_cubes = copy.deepcopy(dataset_cubes_train)
    datetimes_list = copy.deepcopy(datetimes_train)
    flags = copy.deepcopy(flags_train)
    
    return datetimes_list, dataset_cubes, flags


class HimawariDataset(torch.utils.data.Dataset):
    """ Create Dataset object withs data, labels and transformations """
    
    def __init__(self, dataset_cubes, flags, transform=None, target_transform=None):
        self.img_labels = flags
        self.img_cubes = dataset_cubes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = np.moveaxis(self.img_cubes[idx], 0, -1)
        label = int(self.img_labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



def load_loaders(train_list, test_list, parent_dir, event_id):
    """ Create training and testing data loaders """

    train_datetimes_list, train_dataset_cubes, train_flags = load_data(train_list, parent_dir, event_id)
    test_datetimes_list, test_dataset_cubes, test_flags = load_data(test_list, parent_dir, event_id)

    # Transform values genrated from means and variances of training set
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((3.30343164e-01, 2.98926532e-01, 3.94711499e-01, 3.02370269e+02, 2.84893006e+02, 2.67080650e+02), 
                          (0.03862246, 0.04021998, 0.04149022, 3.93229366, 3.71602946,2.76244515))])

    batch_size = 64

    trainset = dp.HimawariDataset(dataset_cubes=np.array(train_dataset_cubes), flags=train_flags, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = dp.HimawariDataset(dataset_cubes=test_dataset_cubes, flags=test_flags, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def load_rfdata(train_list, test_list, parent_dir, event_id):
    """ Load data and calculates statistics for RF model """

    xtrain, ytrain, xtest, ytest = [1,1,1,1,]
    
    return xtrain, ytrain, xtest, ytest