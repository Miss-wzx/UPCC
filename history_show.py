from history.history import History
import pickle
import matplotlib.pyplot as plt

# h = pickle.loads(open('./history/folding_net_e1000_2k.pkl', 'rb').read())
# h = pickle.loads(open('./history/paun_7k_e1000.pkl', 'rb').read())
h = pickle.loads(open('./history/paun_e1000.pkl', 'rb').read())
# h = pickle.loads(open('./history/paun_e500_2k.pkl', 'rb').read())

# h = pickle.loads(open('./history/pcn_e1000_2k.pkl', 'rb').read())
h = pickle.loads(open('./history/paun_e1000_1155.pkl', 'rb').read())
# h = pickle.loads(open('./history/paun_e1000_2k.pkl', 'rb').read())
# h = pickle.loads(open('./history/folding_net_e1000_1155.pkl', 'rb').read())

# h = pickle.loads(open('./history/pcn_e1000_1155.pkl', 'rb').read())
h = pickle.loads(open('./history/paun_e1000_2k_1.pkl', 'rb').read())

# h = pickle.loads(open('./history/paun_e1000_lw_1155.pkl', 'rb').read())
# h = pickle.loads(open('./history/folding_net_e1000.pkl', 'rb').read())
# h = pickle.loads(open('./history/pcn_e1000.pkl', 'rb').read())
# h = pickle.loads(open('./history/paun_e1000_lw_1155_11.pkl', 'rb').read())
# h = pickle.loads(open('./history/paun_e1000_lw_1155_2.pkl', 'rb').read())
h = pickle.loads(open('./history/paun_e1000_lw_1155_15.pkl', 'rb').read())
# h = pickle.loads(open('./history/pcn_e1000_1155_1.pkl', 'rb').read())

print(len(h.train_loss))
print(h.train_loss)

plt.plot(h.train_loss, label='train_loss')

plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.legend()
plt.show()

# print(h.train_lr)
# plt.plot(h.train_lr, label='lr')
# plt.xlabel('epoch')
# plt.ylabel('lr')
# plt.legend()
# plt.show()
#
# print(h.test_loss)
# plt.plot(h.test_loss, label='test_loss')
# plt.xlabel('epoch')
# plt.ylabel('test_loss')
# plt.legend()
# plt.show()
#
# # plt.plot(h.test_cd, label='test_cd')
# # plt.xlabel('epoch')
# # plt.ylabel('test_cd')
# # plt.legend()
# # plt.show()
# #
# # plt.plot(h.test_emd, label='test_emd')
# # plt.xlabel('epoch')
# # plt.ylabel('test_emd')
# # plt.legend()
# # plt.show()
# #
# # plt.plot(h.test_fs, label='test_fs')
# # plt.xlabel('epoch')
# # plt.ylabel('test_fs')
# # plt.legend()
# # plt.show()
#
