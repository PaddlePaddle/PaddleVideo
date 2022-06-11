import http from '@/tools/axios-http-tool.js'

export default async function loadDict() {
	const DICT = {}
	const res = await http.post('/dict/list')
	for (let tableKey in res.data.data) {
		const tableItem = res.data.data[tableKey]
		for (let fieldKey in tableItem) {
			let itemArr = tableItem[fieldKey]
			buildDict(DICT, tableKey, fieldKey, itemArr)
		}
	}
	console.log(DICT)
	return DICT
}

function buildDict(DICT, tableKey, fieldKey, item) {
	if (DICT[`${tableKey.toUpperCase()}`] == null) {
		DICT[`${tableKey.toUpperCase()}`] = {}
	}
	if (DICT[`${tableKey.toUpperCase()}`][`${fieldKey.toUpperCase()}`] == null) {
		DICT[`${tableKey.toUpperCase()}`][`${fieldKey.toUpperCase()}`] = {}
	}
	DICT[`${tableKey.toUpperCase()}`][`${fieldKey.toUpperCase()}`] = function (key, defaultValue = '') {
		if (key == null || item[key] == null) {
			return defaultValue
		}
		return item[key].dictValue || defaultValue
	}
}
