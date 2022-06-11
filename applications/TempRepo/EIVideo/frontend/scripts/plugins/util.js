import {parse} from 'querystring'
import {URL} from 'url'

export const queryRE = /\?.*$/
export const hashRE = /#.*$/

export const cleanUrl = (url) =>
  url.replace(hashRE, '').replace(queryRE, '')

export function parseRequest(id) {
  const {search} = new URL(id, 'file:')
  if (!search) {
    return {}
  }
  return parse(search.slice(1))
}
