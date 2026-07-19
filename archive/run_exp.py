from trainer2 import ActorTrainer
import gymnasium as gym
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="CartPole-v1")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--steps_per_iter", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_ep_len", type=int, default=500)
    p.add_argument("--explore", type=str, default="temperature+m-exclude",
                   help="temperature+m-exclude | temperature | m-exclude | none")
    p.add_argument("--max_m_exclude", type=int, default=2)
    p.add_argument("--base_temp", type=float, default=1.0)
    args = p.parse_args()

    trainer = ActorTrainer(
        env_id=args.env_id,
        seed=args.seed,
        beta=args.beta,
        G=args.G,
        explore=args.explore,
        max_m_exclude=args.max_m_exclude,
        lr=args.lr,
        steps_per_iter=args.steps_per_iter,
        iters=args.iters,
        max_ep_len=args.max_ep_len,
    )
    trainer.train()

if __name__ == "__main__":
    main()
