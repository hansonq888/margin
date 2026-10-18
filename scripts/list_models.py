"""List models available on your Nebius Token Factory account (find the real Nemotron IDs)."""

from margin.nebius import Nebius


def main() -> None:
    client = Nebius().client
    ids = sorted(m.id for m in client.models.list())
    nemotron = [i for i in ids if "nemotron" in i.lower()]
    print(f"{len(ids)} models available.\n")
    print("Nemotron models:")
    for i in nemotron or ["  (none found)"]:
        print(f"  {i}")
    print("\nAll models:")
    for i in ids:
        print(f"  {i}")


if __name__ == "__main__":
    main()
