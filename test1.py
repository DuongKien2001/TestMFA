from config import cfg
cfg.merge_from_file("configs/mfa.yml")
print(cfg.INPUT.TARGET)
a = ''
if a:
    print(1)
else:
    print(2)