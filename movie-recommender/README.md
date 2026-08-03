# Getting Started with Create React App

## TEARS quality recommender setup

This frontend sends both the TEARS and GERS recommendation requests to the
quality-improved EASE API. Copy `.env.example` to `.env` and set the API base
URL (without `/recommend`):

```env
REACT_APP_QUALITY_API_URL=http://127.0.0.1:8001
```

Start the backend from the `tears_project_final` directory:

```bash
. .venv/bin/activate
uvicorn quality_api:app --host 127.0.0.1 --port 8001
```

Then start this frontend:

```bash
npm install
npm start
```

When the frontend runs on Windows and the API runs on a remote Mila machine,
either set `REACT_APP_QUALITY_API_URL` to the API's HTTPS address or create an
SSH tunnel from Windows before starting the frontend:

```powershell
ssh -L 8001:127.0.0.1:8001 your-mila-host
```

Create React App reads environment variables when it starts, so restart
`npm start` after changing `.env`.

## Deferred genre exclusion UI

The "Genres to exclude" input is intentionally hidden from the TEARS interface.
Its state and API request field remain in the code so we can complete and restore
the feature later without changing the current recommendation flow.

## Catalog and recommendation display behavior

- The summary textarea uses `h-64`, twice its previous default height.
- Verified TMDB metadata carries `vote_average` into onboarding catalog cards,
  preventing known movies such as *The Godfather (1972)* from displaying an
  unavailable rating.
- TMDB lookup adds a conservative variant with MovieLens parenthetical aliases
  removed. For example, *Independence Day (ID4) (1996)* is searched as both
  `Independence Day (ID4)` and `Independence Day`, while release-year verification
  remains mandatory.
- Catalog and recommendation poster images fall back to
  `/placeholder_poster.png` when metadata has no poster or the image request
  fails. This behavior is applied to both TEARS and GERS recommendation cards.

The live React process runs behind the stable Tailscale Funnel documented in
`../docs/TEARS_CATALOG_EXCLUSION_AND_PUBLIC_DEPLOYMENT.md`. Changes on shared
storage may not trigger the development server's file watcher across Slurm
nodes, so restart the `frontend` tmux pane after deploying UI changes.

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
