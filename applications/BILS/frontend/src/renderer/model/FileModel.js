import { useService } from '../composables'
const { getFileName, isFolder } = useService('FileService')
export default async function getFileModel (path) {
  return {
    name: await getFileName(path),
    type: await isFolder(path) ? 'folder' : 'file',
    path: path
  }
}
