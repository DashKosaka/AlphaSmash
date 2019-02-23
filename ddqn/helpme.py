for e in range(EPISODES):
    done = False
    score = 0

    history = np.zeros([5, 84, 84], dtype=np.uint8)
    step = 0
    state = env.reset()
    life = number_lives

    get_init_state(history, state)
    print('starting')
    while not done:
        step += 1
        frame += 1

        # Select and perform an action
        action = agent.get_action(np.float32(history[:4, :, :]) / 255.)

        # Get the next frame
        next_state, reward, done, info = env.step(action + 1)

        # Set the most recent history = new frame
        frame_next_state = get_frame(next_state)
        history[4, :, :] = frame_next_state

        # Check if agent is still alive
        terminal_state = check_live(life, info['ale.lives'])

        # Store the transition in memory 
        agent.memory.push(deepcopy(frame_next_state), action, r, terminal_state)
        # Start training after random sample generation
        if(frame >= train_frame):
            agent.train_policy_net(frame)
            # Update the target network
            if(frame % Update_target_network_frequency)== 0:
                agent.update_target_net()
        if frame % 100 == 0:
            print(score)
        print(score)
        history[:4, :, :] = history[1:, :, :]





        if frame % 50000 == 0:
            print('now time : ', datetime.now())
            rewards.append(np.mean(evaluation_reward))
            episodes.append(e)
            pylab.plot(episodes, rewards, 'b')
            pylab.savefig("./save_graph/breakout_dqn.png")

        if done:
            evaluation_reward.append(score)
            # every episode, plot the play time
            print("episode:", e, "  score:", score, "  memory length:",
                  len(agent.memory), "  epsilon:", agent.epsilon, "   steps:", step,
                  "    evaluation reward:", np.mean(evaluation_reward))

            # if the mean of scores of last 10 episode is bigger than 400
            # stop training
            if np.mean(evaluation_reward) > 10:
                torch.save(agent.policy_net, "./save_model/breakout_dqn_test.pth")
                sys.exit()