import {parentPort, workerData} from 'worker_threads'

const port = parentPort
if (!port) throw new Error('IllegalState')

port.on('message', () => {
  port.postMessage(`hello ${workerData}`)
})
