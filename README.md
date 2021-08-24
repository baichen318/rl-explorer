# design-explorer
Design Explorer: Design Space Explorer focusing on the CPU design

## End-to-end Flow
```bash
$ python main.py -c configs/rl-explorer.yml
```

## TODO
- VLSI:
	* Log each running compilation
	* Re-run when compilation occurs errors
	* auto-vlsi.sh needs to handle error compilations
- Invalid configs.:
    * Assign a large negtive reward
- Transition:
    * Record transitions in the buffer
