import os
import random
from argparse import ArgumentParser

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def get_model():
    return nn.Sequential(nn.Linear(4, 30),
                         nn.LeakyReLU(),
                         nn.Linear(30, 30),
                         nn.LeakyReLU(),
                         nn.Linear(30, 2))


class EnvironmentInteractionsDataset(Dataset):
    def __init__(self, samples, ):
        self.samples = self.transform(samples)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def transform(self, samples):
        transformed_samples = []
        for (old_state, action, reward, new_state) in samples:
            old_state_tensor = torch.FloatTensor(old_state, )
            new_state_tensor = torch.FloatTensor(new_state)
            reward = torch.tensor(reward)
            transformed_samples.append((old_state_tensor, action, reward, new_state_tensor))
        return transformed_samples


def test(env, model, max_steps, device):
    env.reset()
    model.eval()
    observation = env.reset()
    for t in range(max_steps):
        env.render()
        input_state = torch.tensor(observation).float().to(device)
        action = torch.argmax(model(input_state)).item()
        observation, reward, done, info = env.step(action)


def sample_from_environment(env, model, epoch, max_steps, device, random_action_prob=0.8, angle_reward_weight=0.5,
                            render=False):
    """
    Сэмплирование примеров взаимодействия модели со средой, а также
    результатов случайных разведочных действий
    """
    train_samples = []
    num_steps_alive = []
    env.reset()
    model.eval()
    for i_episode in range(epoch):
        observation_new = env.reset()
        observation_old = env.reset()
        for t in range(max_steps):
            if render:
                env.render()
            # С некоторой вероятностью совершаем случайное действие
            if random.random() > 1 - random_action_prob:
                action = env.action_space.sample()
            # Получаем следующее действие из модели
            else:
                input_state = torch.tensor(observation_new).float().to(device)
                action = torch.argmax(model(input_state)).item()
            observation_new, reward, done, info = env.step(action)
            if t > 0:
                # Награда за маленькое отклонение маятника от вертикали (угол)
                if abs(observation_new[2]) < 0.2:
                    angle_reward = 0.25
                elif abs(observation_new[2]) < 0.15:
                    angle_reward = 0.5
                elif abs(observation_new[2]) < 0.1:
                    angle_reward = 0.75
                else:
                    angle_reward = 0
                # Награда за малое отклонение машины от начальной точки, т.е. от центра,
                # т.е. за большое расстояние до краёв среды
                if abs(observation_new[0]) < 4.:
                    position_reward = 0.25
                elif abs(observation_new[0]) < 2.:
                    position_reward = 0.5
                elif abs(observation_new[0]) < 1.:
                    position_reward = 0.75
                else:
                    position_reward = 0
                # Взвешиваем награды
                reward += angle_reward_weight * angle_reward + (1 - angle_reward_weight) * position_reward
                # Запоминаем тренировочный пример
                each_sample = (observation_old, action, reward, observation_new)
                train_samples.append(each_sample)

            observation_old = observation_new

            if done:
                num_steps_alive.append(t + 1)
                break
    avg_game_length = sum(num_steps_alive) / len(num_steps_alive)
    print(f"Avg. game length: {avg_game_length} steps")
    return train_samples, avg_game_length


def train(target_model, eval_model, trainloader, criterion, optimizer, device, num_epochs, gamma):
    """
    Обучение модели на одном наборе примеров о взаимодействии со средой. Обучение основано на материалах
    видеокурса Stanford'а (youtu.be/lvoHnicueoE, cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf).
    В частности, используется модель Беллмана.
    """
    train_losses = []
    iter_times = 0
    target_model.eval()
    eval_model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            if iter_times % 100 == 0:
                target_model.load_state_dict(eval_model.state_dict())
            old_state, action, reward, new_state = data
            old_state = old_state.to(device)
            new_state = new_state.to(device)
            reward = reward.to(device)
            action = action.to(device)
            optimizer.zero_grad()

            q_t0 = eval_model(old_state)
            q_t1 = target_model(new_state).detach()
            q_t1 = gamma * (reward + torch.max(q_t1, dim=1)[0])

            loss = criterion(q_t1, torch.gather(q_t0, dim=1, index=action.unsqueeze(1)).squeeze(1))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            iter_times += 1
    target_model.load_state_dict(eval_model.state_dict())
    avg_loss = sum(train_losses) / len(train_losses)
    print(f"Avg. loss: {avg_loss}")
    return avg_loss


def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', default=r"test")
    parser.add_argument('--test_model', required=False, default="ep_167_model.pth")
    parser.add_argument('--models_dir', default=r"models/")
    args = parser.parse_args()
    mode = args.mode
    models_dir = args.models_dir

    # Используется среда для задачи перевернутого маятника от OpenAI:
    # gym.openai.com
    env = gym.make('CartPole-v0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_model, eval_model = get_model().to(device), get_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(eval_model.parameters(), lr=1e-2)
    train_samples = []

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if mode == "test":
        model_name = args.test_model
        model_path = os.path.join(models_dir, model_name)
        eval_model.load_state_dict(torch.load(model_path))
        target_model.load_state_dict(torch.load(model_path))
        test(env, eval_model, 10000, device)
    elif mode == "train":
        # Вес награды за малое отклонение маятника от вертикали
        angle_reward_weight = 0.5
        num_epochs = 200
        avg_losses = []
        avg_game_lengths = []
        for i in range(num_epochs):
            print(f"Epoch: {i + 1}")
            # В начале обучения делаем больше разведочных случайных шагов.
            # К концу обучения больше полагаемся на уже неплохо обученную модель
            random_action_prob = 0.8 - 0.7 * (i / num_epochs)
            # Каждую 10-ю эпоху рендерим, что происходит во время взаимодействия со средой
            sample_times = 10
            if i % 10 == 0:
                render = True
            else:
                render = False
            env_samples, avg_game_length = sample_from_environment(env, eval_model, sample_times, 500, device,
                                                                   random_action_prob,
                                                                   angle_reward_weight, render)
            avg_game_lengths.append(avg_game_length)
            train_samples = env_samples

            if len(env_samples) > sample_times * 160:
                print(f"Epoch {i + 1}, saved model")
                model_path = os.path.join(models_dir, f"ep_{i}_model.pth")
                torch.save(eval_model.state_dict(), model_path)
            train_data_set = EnvironmentInteractionsDataset(train_samples)
            train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=32, shuffle=True, num_workers=0,
                                                            pin_memory=True)
            gamma = np.float(0.9)
            avg_epoch_loss = train(target_model, eval_model, train_data_loader, criterion, optimizer, device, 10, gamma)
            avg_losses.append(avg_epoch_loss)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

        ax[0].plot(avg_losses, label='Train loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_title('Train loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_title('Avg. game length (# steps)')
        ax[1].plot(avg_game_lengths, label='Avg. game length (# steps)')
        plt.legend()
        plt.show()
    env.close()


if __name__ == '__main__':
    main()
