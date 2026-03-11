from torchaudio.datasets import SPEECHCOMMANDS


def main():
    ds = SPEECHCOMMANDS(
        root="data",
        # url='speech_commands_v0.02',
        download=True,
    )
    

if __name__ == "__main__":
    main()
