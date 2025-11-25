// Polyfill dla Buffer (wymagane przez jpeg-js)
import { Buffer } from 'buffer';
if (typeof global !== 'undefined') {
  (global as any).Buffer = Buffer;
} else if (typeof globalThis !== 'undefined') {
  (globalThis as any).Buffer = Buffer;
}

import { registerRootComponent } from 'expo';

import App from './App';

// registerRootComponent calls AppRegistry.registerComponent('main', () => App);
// It also ensures that whether you load the app in Expo Go or in a native build,
// the environment is set up appropriately
registerRootComponent(App);
