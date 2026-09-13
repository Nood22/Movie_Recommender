import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import LandingPage from "./components/LandingPage";
import TearsApp from "./components/TearsApp";
import GersApp from "./components/GersApp";

function WrapperTears() {
  const navigate = useNavigate();
  return <TearsApp goBack={() => navigate("/")} />;
}

function WrapperGers() {
  const navigate = useNavigate();
  return <GersApp goBack={() => navigate("/")} />;
}

export default function App() {
  // The same build is exposed both at the canonical root URL and at /pilot.
  // Select the router base from the incoming URL instead of baking /pilot into
  // every client-side route at build time.
  const basename =
    window.location.pathname === "/pilot" ||
    window.location.pathname.startsWith("/pilot/")
      ? "/pilot"
      : "/";

  return (
    <Router basename={basename}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/tears" element={<WrapperTears />} />
        <Route path="/gers" element={<WrapperGers />} />
      </Routes>
    </Router>
  );
}
