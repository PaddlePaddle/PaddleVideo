async function main(output) {
   output('build_number', process.env.GITHUB_RUN_NUMBER);
}

function setOutput(name, value) {
    process.stdout.write(Buffer.from(`::set-output name=${name}::${value}`))
}

main(setOutput);
