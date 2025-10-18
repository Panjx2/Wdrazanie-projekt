# Wdrażanie projektu

The React Native mobile application lives in the [`CatsApp/`](CatsApp/) directory. All JavaScript dependencies are managed with **npm**, so install packages from inside that folder before running any scripts.

## Getting started

```sh
cd CatsApp
npm install
```

Once the dependencies are in place you can use the built-in npm scripts (each command invokes the local React Native CLI through `npx`):

```sh
# Start Metro
npm run start

# Build and run on Android (emulator or device required)
npm run android

# Build and run on iOS (requires macOS with Xcode)
npm run ios
```

For additional environment and troubleshooting tips, see [`CatsApp/README.md`](CatsApp/README.md).
