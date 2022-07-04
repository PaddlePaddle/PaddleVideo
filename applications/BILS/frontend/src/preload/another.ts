import {readFile} from 'fs-extra'

console.log('hello world 2nd preload!')

/**
 * You can access node module here!
 */
export async function readSomeFile() {
  console.log('You can use module module in preload no matter the nodeIntegration!')
  return readFile('/abc')
}
