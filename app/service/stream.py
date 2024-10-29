from .. import globals


def stream():
    while True:
        if globals._frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + globals._frame + b"\r\n"
            )
        else:
            break
