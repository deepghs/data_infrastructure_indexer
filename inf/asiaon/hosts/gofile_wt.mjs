// Computes gofile's website token for a guest account token, by running the site's own generator.
//
// The generator is a third-party obfuscated script, so this file is only ever run under node's
// permission model with no filesystem access beyond the directory holding it and the script, no
// child processes and an empty environment. It reads wt.obf.js from next to itself and prints the token.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [, , accountToken, userAgent] = process.argv;
const ctx = {
    console, performance, Date, Math, TextEncoder, TextDecoder, crypto: globalThis.crypto, setTimeout, clearTimeout, btoa, atob,
    navigator: {userAgent, language: 'en-US', languages: ['en-US', 'en'], platform: 'Linux x86_64'},
    location: {href: 'https://gofile.io/', origin: 'https://gofile.io', hostname: 'gofile.io'},
    document: {cookie: '', createElement: () => ({})},
    localStorage: {getItem: () => null, setItem() {}},
    screen: {width: 1920, height: 1080},
};
ctx.window = ctx;
ctx.self = ctx;
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(here, 'wt.obf.js'), 'utf8'), ctx);
if (typeof ctx.generateWT !== 'function') {
    console.error('NO_GENERATOR');
    process.exit(2);
}
console.log(await ctx.generateWT(accountToken));
