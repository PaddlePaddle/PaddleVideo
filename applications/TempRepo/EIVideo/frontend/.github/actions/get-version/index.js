const { readFileSync } = require("fs");

async function main(output) {
    output('version', JSON.parse(readFileSync('./package.json')).version)
}

function setOutput(name, value) {
    process.stdout.write(Buffer.from(`::set-output name=${name}::${value}`))
}

main(setOutput);
