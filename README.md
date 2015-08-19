Eigenstyle
==========

Principal Component Analysis and Fashion. This is a fork of https://github.com/graceavery/Eigenstyle.

This repo might differ a lot from the original, as I want to make some experiments with it. I won't be sending pull requests for this reason.

Differences from the original repo
----------------------------------

None (yet).

To Use
------

- Find a bunch of images (I used images of dresses from Amazon).
- Put the ones that match your style in the "like" folder, and the others in the "dislike" folder
- In terminal, run 
```bash
python visuals.py
```

Results
-------

You'll see the principal components in the "eigendresses" folder (examples shown are from my dataset; yours will be different).

In the "history" folder, you'll see a known dress being rebuilt from its components.

In the "recreatedDresses" folder, you can see just the end product of this process for different dresses.

In the "notableDresses" folder, you'll see the prettiest dresses, the ugliest dresses, the most extreme dresses (those that had high scores on many components), etc.

In the "createdDresses" folder, you'll find completely new dresses that were made from choosing random values for the principal components.
