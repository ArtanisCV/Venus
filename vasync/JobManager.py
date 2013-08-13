__author__ = 'Artanis'

import Queue
import threading


class Job(object):
    def __init__(self, jobHandler, args=None, callback=None):
        self.jobHandler = jobHandler
        self.args = args
        self.callback = callback


class JobRunner(threading.Thread):
    def __init__(self, jobManager):
        super(JobRunner, self).__init__()
        self.jobManager = jobManager

    def run(self):
        while not self.jobManager.closed:
            job = self.jobManager.popJob()

            if job is not None:
                try:
                    if job.jobHandler is not None:
                        result = job.jobHandler(job.args)
                    else:
                        result = None

                    self.jobManager.notifyJobFinished(job.callback, result)
                except:
                    pass


class JobManager(object):
    def __init__(self, nThread=4):
        self.closed = False
        self.__initThreadPool(nThread)
        self.__initJobQueue()

        for thread in self.threadPool:
            thread.start()

    def __initThreadPool(self, nThread=4):
        self.threadPool = []
        for i in range(nThread):
            self.threadPool.append(JobRunner(self))

    def __initJobQueue(self):
        self.jobQueue = Queue.Queue()

    def __del__(self):
        self.close()

    def close(self):
        self.closed = True

        # Add fake jobs to stop blocking sub-threads
        for i in range(len(self.threadPool)):
            self.jobQueue.put(None)

    def pushJob(self, jobHandler, args=None, callback=None):
        job = Job(jobHandler, args, callback)

        try:
            self.jobQueue.put(job)
        except:
            raise

    def popJob(self):
        try:
            job = self.jobQueue.get()
        except:
            raise

        return job

    # Here we just invoke the callback in the sub-thread.
    # If the callback requires some unshareable variables, this method should be overrided.
    def notifyJobFinished(self, callback, jobResult):
        if callback is not None:
            callback(jobResult)


import tornado.ioloop


class TornadoJobManager(JobManager):
    def notifyJobFinished(self, callback, jobResult):
        tornado.ioloop.IOLoop.current().add_callback(callback, jobResult)