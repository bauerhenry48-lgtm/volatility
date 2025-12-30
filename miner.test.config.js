module.exports = {
  apps: [
    {
      name: "synth_test",
      interpreter: "python3",
      script: "./neurons/miner.py",
      args: "--netuid 247 --logging.debug --logging.trace --subtensor.network test --wallet.name bt --wallet.hotkey test --axon.port 8092 --blacklist.validator.min_stake 0",
      env: {
        PYTHONPATH: ".",
      },
    },
  ],
};
