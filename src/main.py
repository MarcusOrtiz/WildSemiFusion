import subprocess
import argparse
import importlib


def main():
    training_scripts = [
        # "python -m src.training.base --config " + args.config,
        # "python -m src.training.color_expert --config " + args.config,
        # "python -m src.training.semantics_expert --config " + args.config,
        "python -m src.training.color --config " + args.config,
    ]

    failed_scripts = []

    for script in training_scripts:
        try:
            print(f"Starting: {script}")
            subprocess.run(
                script,
                shell=True,
                check=True  # Raise exception if script fails
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: {script} failed with return code {e.returncode}.")
            failed_scripts.append(script)

    # Summary of results
    print("\nTraining Completed.")
    if failed_scripts:
        print("The following scripts failed:")
        for script in failed_scripts:
            print(f"  - {script}")
    else:
        print("All scripts ran successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequentially train all available models")
    parser.add_argument('--config', type=str, default='src.local_config',
                        help='Path to the configuration module for scripts to use(src.local_config | src.aws_config)')
    args = parser.parse_args()

    main()
