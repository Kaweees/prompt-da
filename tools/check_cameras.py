import zenoh
import struct
import time

conf = zenoh.Config()
z = zenoh.open(conf)

def cb(sample):
    h, w = struct.unpack_from("<ii", bytes(sample.payload), 8)
    print(str(sample.key_expr) + ": " + str(w) + "x" + str(h))

topics = ["body/camera/wide", "body/camera/road"]
for t in topics:
    z.declare_subscriber(t, cb)

print("Listening on " + ", ".join(topics) + " ...")
time.sleep(10)
print("Done.")
z.close()
