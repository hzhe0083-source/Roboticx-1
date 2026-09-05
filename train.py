"""Compatibility command for the MetaWorld trainer.

Shared training functions live in va_compound.training, not this entrypoint.
"""
from train_metaworld import main


if __name__ == "__main__":
    main()
