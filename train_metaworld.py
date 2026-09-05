"""MetaWorld DINO / peer-World training entrypoint."""
from va_compound.training.config import parse_args
from va_compound.training.engine import run_metaworld


def main():
    run_metaworld(parse_args())


if __name__ == "__main__":
    main()
