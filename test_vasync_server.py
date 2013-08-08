__author__ = 'Artanis'

import shutil

import tornado.web
import tornado.websocket
import tornado.httpserver

import vasync.JobManager


class FileHandler(tornado.websocket.WebSocketHandler):
    jobManager = vasync.JobManager.TornadoJobManager()

    def open(self):
        self.write_message("Connect with server.")

    def on_close(self):
        pass

    def moveDir(self, dirName):
        shutil.copytree(r"D:\dataset" + '\\' + dirName, r"D:\tmp" + '\\' + dirName)
        return dirName

    def finish(self, dirName):
        self.write_message("Finish Moving " + dirName)

    def on_message(self, dirName):
        if dirName == "close":
            FileHandler.jobManager.close()
        else:
            FileHandler.jobManager.pushJob(self.moveDir, dirName, self.finish)


def main():
    application = tornado.web.Application([
        ('/file', FileHandler)
    ])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8000)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()