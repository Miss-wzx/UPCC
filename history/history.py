import datetime
import pickle


class History:
    def __init__(self, model_name, path, epoch, batch_size, info=''):
        self.model_name = model_name
        self.train_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.save_history_path = path
        self.train_loss = []
        self.test_loss = []
        self.test_cd = []
        self.test_emd = []
        self.test_fs = []
        self.train_lr = []
        self.train_epoch = epoch
        self.train_batch_size = batch_size
        self.info = info

    def save_history(self):
        path = self.save_history_path+'/'+self.model_name+'_e{}'.format(self.train_epoch)+self.info+'.pkl'
        with open(path, "wb") as save_file:
            save_file.write(pickle.dumps(self))
            save_file.close()
        return path

    def print_info(self):
        print(self.model_name)


if __name__ == '__main__':
    h = History('model', './', 100, 16)
    print(h.train_time)
    sp = h.save_history()
    rp = pickle.loads(open(sp, 'rb').read())
    print(rp.model_name)
    rp.print_info()
