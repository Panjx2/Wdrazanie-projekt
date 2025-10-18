This project is built with [**React Native**](https://reactnative.dev) and uses **npm** for JavaScript dependency management. The local CLI is provided by [`@react-native-community/cli`](https://github.com/react-native-community/cli), so the `npm run …` scripts (or the equivalent `npx react-native …` commands) always invoke the correct binaries.

# Environment prerequisites

- Install **Node.js 20** or newer.
- Install the native SDKs required for your platform (Android Studio with SDK/NDK level 36, or Xcode with CocoaPods via Bundler).

# Installing dependencies

```sh
cd CatsApp
npm install
```

The command above creates the usual `node_modules` tree and hoists the React Native CLI into `node_modules/.bin`, making `npx react-native …` available without any global installs.

# Running the app

You can work either with the npm scripts or run the CLI via `npx`:

```sh
# Start Metro
npm run start
# or: npx react-native start

# Launch Android (requires a running emulator or connected device)
npm run android
# or: npx react-native run-android

# Launch iOS (requires macOS with Xcode)
npm run ios
# or: npx react-native run-ios
```

# iOS native dependencies

For iOS development, install the Ruby toolchain once and then resolve CocoaPods through Bundler:

```sh
bundle install
cd ios
bundle exec pod install
```

Re-run `bundle exec pod install` whenever native dependencies change.

# Troubleshooting

- `npx @react-native-community/cli doctor` can help diagnose common environment issues.
- If installs fail, clear any previous `node_modules` folder and re-run `npm install` to regenerate the lockfile.

# Learn more

- [React Native Docs](https://reactnative.dev/docs/getting-started)
- [Android Environment Setup](https://reactnative.dev/docs/environment-setup?os=android)
- [iOS Environment Setup](https://reactnative.dev/docs/environment-setup?os=ios)
