import torch
from torch.autograd import Variable
from utils import get_args, Config
from preprocess import xor_data_generate
from train import train_step
from test import test_C
import models


def main():
    torch.manual_seed(233)
    torch.cuda.set_device(0)
    args = get_args()
    print("generating config")
    config = Config(
        state_dim=args.hidden,
        input_dim=args.input_dim,
        hidden=args.hidden,
        output_dim=args.num_classes,
        epsilon=args.epsilon
    )
    gamma = args.gamma
    reward_amplify = args.reward_amplify
    passive_drive = args.passive_drive
    memory = models.Memory(args.capacity)
    m = args.batch_size
    print("initializing networks")
    E = models.Shared_Encoder(config)
    Q = models.Simple_Q_Net(config)    # 2-dim x-or problem
    Q_t = models.Simple_Q_Net(config)
    Q_t.load_state_dict(Q.state_dict())     # let Q and Q_t be identical initially

    C = models.SimpleNNClassifier(config)

    episode_length = args.episode_length
    episode_number = args.episode_number

    print("initializing optimizers")
    optimizer_E = torch.optim.Adam(E.parameters(), lr=args.lr, betas=(0., 0.999))
    optimizer_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0., 0.999))
    optimizer_Q = torch.optim.Adam(Q.parameters(), lr=args.lr, betas=(0., 0.999))

    # enable gpu
    E.cuda()
    C.cuda()
    Q.cuda()
    Q_t.cuda()

    #test_C(C, E)
    loss_last = Variable(torch.tensor([0.])).cuda()
    X_eval, Y_eval = xor_data_generate(args.eval_set_size)
    X_eval = X_eval.cuda()
    Y_eval = Y_eval.cuda()
    for i in range(episode_number):
        #X_eval, Y_eval = xor_data_generate(args.eval_set_size)
        #X_eval = X_eval.cuda()
        #Y_eval = Y_eval.cuda()
        X, Y = xor_data_generate(m)
        X = X.cuda()
        Y = Y.cuda()
        for t in range(episode_length):
            try:
                X, Y, loss_last, reward = train_step(E=E,
                                                     C=C,
                                                     Q=Q,
                                                     Q_t=Q_t,
                                                     X=X,
                                                     Y=Y,
                                                     eval_X=X_eval,
                                                     eval_Y=Y_eval,
                                                     gamma=gamma,
                                                     loss_last=loss_last,
                                                     memory=memory,
                                                     optimizer_C=optimizer_C,
                                                     optimizer_E=optimizer_E,
                                                     optimizer_Q=optimizer_Q,
                                                     reward_amplify=reward_amplify,
                                                     passive_drive=passive_drive)
                print("Episode %i step %i, loss=%f, reward=%f" % (
                    i, t, loss_last.detach().cpu().numpy(), reward.detach().cpu().numpy()))
            except Exception as e:
                print("Cannot train the model on this step, error:", e)

        Q_t = Q
        if i % 20 == 0:
            test_C(C, E)
            state = {
                'E_state_dict': E.state_dict(),
                'E_optimizer': optimizer_E.state_dict(),
                'C_state_dict': C.state_dict(),
                'C_optimizer': optimizer_C.state_dict(),
                'Q_state_dict': Q.state_dict(),
                'Q_optimizer': optimizer_Q.state_dict(),

            }
            model_name = "cog396test_main_episode_" + str(i) + ".tr"
            torch.save(state, model_name)


if __name__ == '__main__':
    main()
    print("hello")





